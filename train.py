from tensorflow.keras.optimizers import Adam , SGD , RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from utils import get_data
import baseNet as nn
import pprint
import random

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

