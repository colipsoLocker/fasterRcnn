from tensorflow.keras.optimizers import Adam , SGD , RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from utils import get_data , get_anchor_gt , calc_iou , rpn_to_roi
import baseNet as nn
import pprint
import random
import config
import losses
import numpy as np
import time



all_imgs, classes_count, class_mapping = get_data()

if 'bg' not in classes_count: #补充背景类型
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

inv_map = {v: k for k, v in class_mapping.items()} #图像序号转换{序号：class_name}

pprint.pprint(classes_count)
pprint.pprint(class_mapping)


random.shuffle(all_imgs) #随机化

num_imgs = len(all_imgs)

#分开成训练集和检验集
#[{filename:{filepath:*,width:*,height:*,'imageset': 'trainval',bboxes:['class':*, 'x1': *, 'x2': *, 'y1':*, 'y2':*]}}]
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

# 输入的train_imgs 的格式[{filename:{filepath:*,width:*,height:*,bboxes:['class':*, 'x1': *, 'x2': *, 'y1':*, 'y2':*]}}]
# 返回的格式 特征层 (output_height, output_width, num_anchors) (output_height, output_width, num_anchors*4)
#np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug
data_gen_train = get_anchor_gt(train_imgs, classes_count,  nn.get_img_output_length, mode='train')
data_gen_val = get_anchor_gt(val_imgs, classes_count, nn.get_img_output_length, mode='val')

#模型输入结构
input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))


#定义基础网络
shared_layers = nn.nn_base(img_input, trainable=True) #基本网络的共享特征层输出

#定义rpn网络
num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)  # [x_class, x_regr, base_layers]

#定义分类网络
classifier = nn.classifier(shared_layers, roi_input, config.num_rois, nb_classes=len(classes_count), trainable=True)


#构建模型
model_rpn = Model(img_input, rpn[:2]) # rpn[:2]：[x_class, x_regr】
model_classifier = Model([img_input, roi_input], classifier)
# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier) 
#是为了保存两个模型使用的,需要注意的是list+list是列表连接运算符

#定义优化方法
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)

#编译模型
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')


#手工定义训练阶段
epoch_length = 1000
num_epochs = int(config.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5)) #之所以是5是因为有个分类的loss，还有四个回归的loss
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True


for epoch_num in range(num_epochs):

	progbar = Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:
			#每个迭代处理一次rpn_accuracy_rpn_monitor用来监控表现
			if len(rpn_accuracy_rpn_monitor) == epoch_length and config.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train) #np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug 用来给分类网络的输入

			loss_rpn = model_rpn.train_on_batch(X, Y) #训练model_rpn

			P_rpn = model_rpn.predict_on_batch(X) #测试rpn

			#R:boxes, probs  返回经过npm后剩下的bbox以及对应的probs
			R = rpn_to_roi(P_rpn[0], P_rpn[1],  use_regr=True, overlap_thresh=0.7, max_boxes=300)
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = calc_iou(R, img_data, class_mapping )

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))
			#看起来是均衡选择样本
			if config.num_rois > 1:
				if len(pos_samples) < config.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, config.num_rois//2, replace=False).tolist()
				try:
					selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples
			else:
				# in the extreme case where num_rois = 1, we pick a random pos or neg sample
				selected_pos_samples = pos_samples.tolist()
				selected_neg_samples = neg_samples.tolist()
				if np.random.randint(0, 2):
					sel_samples = random.choice(neg_samples)
				else:
					sel_samples = random.choice(pos_samples)
			#训练分类网络，得到分类和bbox回归
			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if config.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if config.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save('./models/model_all.h5')

				break
#上面大段的代码都是在写如何计算损失函数。
		except Exception as e:
			print('Exception: {}'.format(e))
			continue

model_all.save('./models/model_all.h5')
model_classifier.save('./models/model_classifier.h5')
model_rpn.save('./models/model_rpn.h5')

print('Training complete, exiting.')
