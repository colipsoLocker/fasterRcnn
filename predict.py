import cv2
from tensorflow.keras.models import load_model
import json
import pprint
import baseNet as nn
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import config
from RoiPoolingConv import RoiPoolingConv
import losses
from PIL import Image
from PIL import ImageDraw
import numpy as np
from utils import get_new_img_size , rpn_to_roi ,apply_regr , non_max_suppression_fast
import time
from progress.bar import Bar

imgFile = './data/predict/car3.jpg'
imgFile = './data/predict/dogs.jpg'
imgFile = './data/predict/1.jpg'
bbox_threshold = 0.8

with open('./models/inv_map.json') as f:
    inv_map = json.load(f)
    pprint.pprint(inv_map)

with open('./models/classes_count.json') as f:
    classes_count = json.load(f)
    pprint.pprint(classes_count)

with open('./models/class_mapping.json') as f:
    class_mapping = json.load(f)
    pprint.pprint(class_mapping)

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
pprint.pprint(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}



num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

model_all = load_model('./models/model_all.h5',
    custom_objects={
        'RoiPoolingConv': RoiPoolingConv , 
        'rpn_loss_cls':losses.rpn_loss_cls , 
        'rpn_loss_regr':losses.rpn_loss_regr ,  
        'class_loss_cls':losses.class_loss_cls ,
        'class_loss_regr':losses.class_loss_regr
                    }
                    )
model_rpn = load_model('./models/model_rpn.h5',
    custom_objects={
        'RoiPoolingConv': RoiPoolingConv , 
        'rpn_loss_cls_fixed_num':losses.rpn_loss_cls(num_anchors) , 
        'rpn_loss_regr_fixed_num':losses.rpn_loss_regr(num_anchors)
        }
        )
model_classifier = load_model('./models/model_classifier.h5',
    custom_objects={
        'RoiPoolingConv': RoiPoolingConv , 
        'class_loss_cls':losses.class_loss_cls ,
        'class_loss_regr_fixed_num':losses.class_loss_regr(len(classes_count)-1)
        }
        )

print("Models are ready!")
st = time.time()
def getOneImage(imageFIle = './predict/1.jpg'):
    img_min_side = float(config.im_size)
    img = Image.open(imageFIle)#根据路径读取图像
    x_img = np.array(img)
    (width, height, _) = x_img.shape
    (resized_width, resized_height) = get_new_img_size(width, height, config.im_size)
    x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    x_img = x_img.astype(np.float32)
    x_img[:, :, 0] -= config.img_channel_mean[0]
    x_img[:, :, 1] -= config.img_channel_mean[1]
    x_img[:, :, 2] -= config.img_channel_mean[2]
    x_img /= config.img_scaling_factor

    x_img = np.transpose(x_img, (2, 0, 1)) #颜色， 长 ， 宽
    x_img = np.expand_dims(x_img, axis=0)
    x_img = np.transpose(x_img, (0, 2, 3, 1)) # 0 ， 长， 宽 ， 颜色
    if width <= height:
        ratio = img_min_side/width
    else:
        ratio = img_min_side/height

    return x_img , ratio ,img

def get_real_coordinates(ratio, x1, y1, x2, y2):

    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2 ,real_y2)


X , ratio , img = getOneImage(imgFile) #img 是PIL格式
[Y1, Y2] = model_rpn.predict(X) #RPN输出区域预测，包括区域以及坐标

pprint.pprint(Y1)
pprint.pprint(Y2)

R = rpn_to_roi(Y1, Y2,  overlap_thresh=0.7) #boxes  #返回经过npm后剩下的bbox以及对应的probs （（左上，右下坐标），序号）    （anchors， 序号 ）
# convert from (x1,y1,x2,y2) to (x,y,w,h)
R[:, 2] -= R[:, 0]
R[:, 3] -= R[:, 1]

print("Caculated ROIS")
draw = ImageDraw.Draw(img)

bboxes = {}
probs = {}

pprint.pprint(R)

bar = Bar('Caculateing class and BBOX', max=R.shape[0]//config.num_rois + 1)

for jk in range(R.shape[0]//config.num_rois + 1):
    bar.next()
    ROIs = np.expand_dims(R[config.num_rois*jk:config.num_rois*(jk+1), :], axis=0) #处理感兴趣的区域
    if ROIs.shape[1] == 0:
        break

    if jk == R.shape[0]//config.num_rois:
        #pad R
        curr_shape = ROIs.shape
        target_shape = (curr_shape[0],config.num_rois,curr_shape[2])
        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
        ROIs_padded[:, :curr_shape[1], :] = ROIs
        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
        ROIs = ROIs_padded
    #从此处可以看出，在训练的时候model_classifier 的input_roi是从训练集中另外输入的。
    #而在预测的时候，input_roi是从rpn网络得到的
    [P_cls, P_regr] = model_classifier.predict([X, ROIs])
    #print("Get final classes and regrs for one batch(config.num_rois)!")
    pprint.pprint(P_cls)
    pprint.pprint(P_regr)
    pprint.pprint(bboxes)
    pprint.pprint(probs)

    for ii in range(P_cls.shape[1]):

        if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            continue

        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]#输出分类网络结论

        if cls_name not in bboxes:
            bboxes[cls_name] = []
            probs[cls_name] = []

        (x, y, w, h) = ROIs[0, ii, :]

        cls_num = np.argmax(P_cls[0, ii, :])
        try:
            (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
            tx /= config.classifier_regr_std[0]
            ty /= config.classifier_regr_std[1]
            tw /= config.classifier_regr_std[2]
            th /= config.classifier_regr_std[3]
            x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)#把归回后的偏移计算上去
        except:
            pass
        #记录原图bbox的坐标位置以及概率值，是个列表，一个类别可能有多个位置
        bboxes[cls_name].append([config.rpn_stride*x, config.rpn_stride*y, config.rpn_stride*(x+w), config.rpn_stride*(y+h)])
        probs[cls_name].append(np.max(P_cls[0, ii, :]))

bar.finish()

all_dets = []
#开始画图
print("Begin to draw the result!")

for key in bboxes: #对每个类别
    bbox = np.array(bboxes[key])
    #做一下非最大值抑制，对多个重叠的框保留最好的一个https://www.cnblogs.com/makefile/p/nms.html。感觉这里还可以优化并加速
    new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
    for jk in range(new_boxes.shape[0]):
        (x1, y1, x2, y2) = new_boxes[jk,:]
        #从特征图映射回原真实坐标，用rpn和roi的转换也估计是映射使用以及论文中偏移计算使用
        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
        print(real_x1, real_y1, real_x2, real_y2)
        #画框框
        draw.rectangle(real_x1, real_y1, real_x2, real_y2, outline=(int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])))

        textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
        all_dets.append((key,100*new_probs[jk]))
        draw.text((real_x1, real_y1-10),textLabel )

        #(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
        #textOrg = (real_x1, real_y1-0)
        #写文字
        #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
        #cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
        #cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

print('Elapsed time = {}'.format(time.time() - st))
print(all_dets)
#cv2.imshow('img', img)
#cv2.waitKey(0)
img.show()