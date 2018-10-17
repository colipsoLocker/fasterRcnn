from tensorflow.keras.optimizers import Adam , SGD , RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from utils import get_data , get_anchor_gt
import baseNet as nn
import pprint
import random
import config


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

