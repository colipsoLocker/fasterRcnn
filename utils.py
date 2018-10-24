import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from PIL import ImageDraw
import pprint
import random
import copy
import config
import itertools
import pdb
import numpy as np
import math
import pprint
import time


def get_data(data_path = './data'):
    '''从voc2007的文件中读取数据
    @data_path: ‘./data’
    @output:
        all_imgs[{filename:{filepath:*,width:*,height:*,'imageset': 'trainval',bboxes:['class':*, 'x1': *, 'x2': *, 'y1':*, 'y2':*]}}]
        classes_count： {classname:对应的数量，}
        class_mapping：{class_name:序号}
    '''

    all_imgs = []

    classes_count = {}

    class_mapping = {}

    visualise = True

    #data_paths = [os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]


    print('Parsing annotation files')

    #for data_path in data_paths:

    annot_path = os.path.join(data_path, 'Annotations')
    imgs_path = os.path.join(data_path, 'JPEGImages')
    imgsets_path_trainval = os.path.join(data_path,'train', 'ImageSets','Main','trainval.txt')
    imgsets_path_test = os.path.join(data_path,'test', 'ImageSets','Main','test.txt')

    trainval_files = []
    test_files = []
    try:
        with open(imgsets_path_trainval) as f:
            for line in f:
                trainval_files.append(line.strip() + '.jpg')
    except Exception as e:
        print(e)

    try:
        with open(imgsets_path_test) as f:
            for line in f:
                test_files.append(line.strip() + '.jpg')
    except Exception as e:
        if data_path[-7:] == 'VOC2012':
            # this is expected, most pascal voc distibutions dont have the test.txt file
            pass
        else:
            print(e)
    
    annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
    idx = 0
    for annot in annots:
        try:
            idx += 1

            et = ET.parse(annot)
            element = et.getroot()

            element_objs = element.findall('object')
            element_filename = element.find('filename').text
            element_width = int(element.find('size').find('width').text)
            element_height = int(element.find('size').find('height').text)

            if len(element_objs) > 0:
                annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                    'height': element_height, 'bboxes': []}

                if element_filename in trainval_files:
                    annotation_data['imageset'] = 'trainval'
                elif element_filename in test_files:
                    annotation_data['imageset'] = 'test'
                else:
                    annotation_data['imageset'] = 'trainval'

            for element_obj in element_objs:
                class_name = element_obj.find('name').text
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                obj_bbox = element_obj.find('bndbox')
                x1 = int(round(float(obj_bbox.find('xmin').text)))
                y1 = int(round(float(obj_bbox.find('ymin').text)))
                x2 = int(round(float(obj_bbox.find('xmax').text)))
                y2 = int(round(float(obj_bbox.find('ymax').text)))
                difficulty = int(element_obj.find('difficult').text) == 1
                annotation_data['bboxes'].append(
                    {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
            all_imgs.append(annotation_data)

            if visualise and random.randint(0,5000) > 4998:
                img = Image.open(annotation_data['filepath'])
                draw = ImageDraw.Draw(img)
                for bbox in annotation_data['bboxes']:
                    draw.rectangle(
                        (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']),
                         outline=(0, 0, 255)
                         )
                    draw.text(
                        (bbox['x1'], bbox['y1']-10),
                         bbox['class']
                         )
                img.show()

        except Exception as e:
            print(e)
            continue
    return all_imgs, classes_count, class_mapping



def augment(img_data, config = config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)
    img = Image.open(img_data_aug['filepath'])#根据路径读取图像
    img = np.array(img) #注意，虽然PIL获取的图像是RGB，但需要转换成array。cv2是直接处理array的
    if augment:
        rows, cols = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:#水平翻转
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:#垂直翻转
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:#旋转90度
            angle = np.random.choice([0,90,180,270],1)[0]
            if angle == 270:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1,0,2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2        
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img #返回输入的img_data_aug 以及 处理后的图像


def union(au, bu, area_intersection): #两个区域的合并,用来辅助计算iou
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi): #两个区域的交叉区域,用来辅助计算iou
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)

def get_new_img_size(width, height, img_min_side=600):#等比例缩放到最短边是img_min_side
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [b for b in class_count.keys() if class_count[b] > 0] #classes_count： {classname:对应的数量，}
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):
        '''
        为样本均衡抽取
        '''
        class_in_img = False

        for bbox in img_data['bboxes']:

            cls_name = bbox['class']

            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True




def calc_rpn(img_data, width, height, resized_width, resized_height, img_length_calc_function , C = config):

    # 各种anchor
    downscale = float(C.rpn_stride) #下采样
    anchor_sizes = C.anchor_box_scales #不同的ancher尺寸
    anchor_ratios = C.anchor_box_ratios #不同的ancher比例
    num_anchors = len(anchor_sizes) * len(anchor_ratios)	#9种ancher

    # calculate the output map size based on the network architecture
    #如果用vgg，那么rpn网络输出的图像是输入图像缩小16倍
    #实际使用的是 train中的get_img_output_length
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives 初始化感兴趣的区域
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])  #bboxes:['class':*, 'x1': *, 'x2': *, 'y1':*, 'y2':*] 有多少个物体

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int) 
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32) #Intersection over Union
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # get the GT box coordinates, and resize to account for image resizing.GT means ground truth
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width)) #gta :ground truth area
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth

    for anchor_size_idx in range(len(anchor_sizes)): #第几个anchor 对每一种anchor都对特征图上所有点都遍历一遍
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0] #生成ancher在原图上的坐标
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
            
            for ix in range(output_width):					#output_with 是经过基础网络后特征图的宽，对网络输出的特征图开始遍历生成密密麻麻的anchers
                # x-coordinates of the current anchor box	
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2 #形成从特征图还原到原始图的左上，右下x坐标。
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
                
                # ignore boxes that go across image boundaries					
                if x1_anc < 0 or x2_anc > resized_width:
                    continue
                    
                for jy in range(output_height):
                    #遍历特征图上的每个点，以及每个点对应回原图的anchor
                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2 #形成从特征图还原到原始图的左上，右下y坐标。
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # bbox_type indicates whether an anchor should be a target 
                    bbox_type = 'neg' #给一个bbox的默认值

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):
                        
                        # get IOU of the current GT box and the current anchor box 计算ground truth和anchors的iou
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        #如果iou是当前有物体区域中最大的，或者已经达到了最低要求。
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0 #计算ground truth中心点
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc)/2.0 #ancher的中心点
                            cya = (y1_anc + y2_anc)/2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc) #计算论文中的tx,ty,tw,th用于bbox回归,按ancher的比例进行缩放
                            ty = (cy - cya) / (y2_anc - y1_anc) # 也是能适应不同大小的物体原因之一
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                        
                        if img_data['bboxes'][bbox_num]['class'] != 'bg':

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]: #记录回特征图的ancher点
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx] #更新第i个bbox的最佳anchor
                                best_iou_for_bbox[bbox_num] = curr_iou #记录当前iou
                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc] #应该就是rpn输出的内容之一最佳anc的坐标
                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th] #偏移量

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                bbox_type = 'pos' #含有物品
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th) #最佳回归偏移量

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # gray zone between neg and pos
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # turn on or off outputs depending on IOUs
                    if bbox_type == 'neg':
                        #那个点（jy,ix）的第几个ancher（anchor_ratio_idx + n_anchratios * anchor_size_idx）是有效的
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1 
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0 #rpn 与gt没有交叉
                    elif bbox_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        #start:start+4 正好对应一个anchor框的四个回归项
                        y_rpn_regr[jy, ix, start:start+4] = best_regr

                        #所以对特征图上每一个点，都至少会有一个anchor

    # we ensure that every bbox has at least one positive RPN region
    #至少保证每一个object都有一个RPN region

    for idx in range(num_anchors_for_bbox.shape[0]): #对每一个object 都用best_anchor_for_bbox 来作为最佳anchor
        if num_anchors_for_bbox[idx] == 0: #说明object 本应该有一个anchor，但没有
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1: #说明还没有交叉
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                best_anchor_for_bbox[idx,3]] = 1
            start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
            y_rpn_regr[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1)) #把值放在第一个纬度，
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0) #为后续送入网络方便，增加第0纬
    #debug
    print("y_rpn_overlap shape:")
    print(np.shape(y_rpn_overlap))
    #enddebug


    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))#把anchor值放在第一个纬度，jy,ix
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1)) #rpn与GT有交叉，且是个有效的bbox
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1)) #rpn与GT不交叉，且是个有效的bbox，为了派出周边无效的bbox？

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256
    #只选择256个有效的区域出来，这个选择的过程可能会影响小物体的检测
    if len(pos_locs[0]) > num_regions/2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2) #从多出128个的正object中抽样。
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0 #将多出来的置为没有物体
        num_pos = num_regions/2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos) #保持正负样本均衡
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1) #矩阵合并，y_rpn_cls：是否包含类，其前半段是该anchor是否可用
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1) #y_rpn_regr：回归梯度，前半段包含是否有效

    #debug
    print('calc_rpn')
    print(np.shape(y_is_box_valid))
    print(np.shape(y_rpn_overlap))
    print(np.shape(y_rpn_cls))
    print(np.shape(y_rpn_regr))
    #enddebug
    return np.copy(y_rpn_cls), np.copy(y_rpn_regr) #返回 (num_anchors ,output_height, output_width, ) (num_anchors*4 ,output_height, output_width )


def get_anchor_gt(all_img_data, class_count,  img_length_calc_function,  C = config,mode='train'):
    '''
    @input:
        all_img_data:[{filename:{filepath:*,width:*,height:*,'imageset': 'trainval',bboxes:['class':*, 'x1': *, 'x2': *, 'y1':*, 'y2':*]}}]
        img_length_calc_function:   width//16,Heigh//16
    '''
    # The following line is not useful with Python 3.5, it is kept for the legacy
    # all_img_data = sorted(all_img_data)

    sample_selector = SampleSelector(class_count)
    
    pprint.pprint(all_img_data[0])
    print('Begin get_anchor_gt')

    while True:
        #debug
        i = 0
        #enddebug
        if mode == 'train':
            np.random.shuffle(all_img_data)

        for img_data in all_img_data:
            print(time.asctime( time.localtime(time.time()) ) + 'Get anchor begin deal img_data')
            try:
                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # read in image, and optionally add augmentation

                if mode == 'train':
                    img_data_aug, x_img = augment(img_data, C, augment=True) #img_data_aug, img #返回输入的img_data_aug 以及 处理后的图像
                else:
                    img_data_aug, x_img = augment(img_data, C, augment=False)

                #img_data_aug：[{filename:{filepath:*,width:*,height:*,bboxes:['class':*, 'x1': *, 'x2': *, 'y1':*, 'y2':*]}}]
                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size) ##等比例缩放到最短边是img_min_side

                # resize the image so that smalles side is length = 600px 用PIL读图，用cv2来处理是个不错的方法，在test中已验证
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                
                print('Get resized x_img')

                try:
                    #返回 (num_anchors ,output_height, output_width) (num_anchors*4 ,output_height, output_width )
                    y_rpn_cls, y_rpn_regr = calc_rpn(img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                except Exception as e:
                    print( e )
                    continue

                # Zero-center by mean pixel, and preprocess image

                #x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB 不用转了，因为不是用的cv2读图用的是PIL
                #做图像正则化
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1)) #颜色， 长 ， 宽
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling #

                x_img = np.transpose(x_img, (0, 2, 3, 1)) # 0 ， 长， 宽 ， 颜色
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))  # 0 ， 长， 宽 ， 值
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1)) # 0 ， 长， 宽 ， 值
                #debug
                print('yield')
                print(np.shape(x_img) , np.shape(y_rpn_cls) , np.shape(y_rpn_regr))
                #enddebug
                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print('Error2 : ' + e)
                continue



def apply_regr(x, y, w, h, tx, ty, tw, th):
    '''
    根据回归的结果移动框的中心点坐标以及框的长， 宽
    '''
    try:
        cx = x + w/2.
        cy = y + h/2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T): #把偏移量作用到anchor上
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy

		w1 = np.exp(tw.astype(np.float64)) * w
		h1 = np.exp(th.astype(np.float64)) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)
		return np.stack([x1, y1, w1, h1])
	except Exception as e:
		print(e)
		return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
    #非最大值抑制，参考https://www.cnblogs.com/makefile/p/nms.html
    #在同个物体的多个框中，找到概率最大的，再去掉与这个概率最大的框roi超过阈值的框。从剩下的框中找概率最大的，重复上述过程。
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	area = (x2 - x1) * (y2 - y1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int

		# find the union
		area_union = area[i] + area[idxs[:last]] - area_int

		# compute the ratio of overlap
		overlap = area_int/(area_union + 1e-6)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs  #返回经过npm后剩下的bbox以及对应的probs




def rpn_to_roi(rpn_layer, regr_layer, use_regr=True, max_boxes=300,overlap_thresh=0.9 ,C = config):
    '''
    @input:
        rpn_layer , regr_layer 对应rpn网络的两个输出，分别是（0 , h, w , anchors）,（0 , h, w , anchors*4）
    '''
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales #[128, 256, 512]
    anchor_ratios = C.anchor_box_ratios # [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

    assert rpn_layer.shape[0] == 1

    (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0

    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3])) #(4,w,h,anchors)

    for anchor_size in anchor_sizes: #128
        for anchor_ratio in anchor_ratios: # [1,1]
            anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride #rpn_stride = 16 映射到特征图上的右下角坐标
            anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride

            regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4] #一个layer是 不同的anchor，最多为9
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols),np.arange(rows)) #对meshgrid的理解https://www.aliyun.com/jiaocheng/516456.html

            A[0, :, :, curr_layer] = X - anchor_x/2 #计算特征图上每个点对应anchor的中心点（可以算到图外）
            A[1, :, :, curr_layer] = Y - anchor_y/2
            A[2, :, :, curr_layer] = anchor_x #对应的anchor高
            A[3, :, :, curr_layer] = anchor_y #对应的anchor宽

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr) #把偏移量作用到anchor上 [x1, y1, w1, h1]

            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])  #去除左上点超出左边界的anchor
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer] #获得右下角坐标
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer]) #左上角坐标不能超过0
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols-1, A[2, :, :, curr_layer]) #右下角坐标不能超过最大
            A[3, :, :, curr_layer] = np.minimum(rows-1, A[3, :, :, curr_layer])

            curr_layer += 1

    all_boxes = np.reshape(A.transpose((0, 3, 1,2)), (4, -1)).transpose((1, 0)) # 变换后就是（（左上，右下坐标），序号）
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1)) #（0 , h, w , anchors） 转为 （anchors， 序号 ）

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0)) #删除不合理的anchor

    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result #boxes, probs  #返回经过npm后剩下的bbox以及对应的probs （（左上，右下坐标），序号）    （anchors， 序号 ）




def calc_iou(R, img_data,  class_mapping, C=config):
    '''
    @input:
        R:rpn_to_roi 函数的输出 boxes, probs  返回经过npm后剩下的bbox以及对应的probs
    '''
    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4)) #每个bbox都有四个坐标决定

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing 折算到resize的图像上
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width))/C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height))/C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height))/C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = [] # for debugging only

    for ix in range(R.shape[0]): #回归框个数
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
                continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0 #中心点
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w) #anchor和gt的差距
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th] #前景
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords)) #背景
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)


    #X：[[x1, y1, w, h],...]  Y1：[classLabel1 ,......] Y1：[[0,0,1,0],....] , [sx*tx, sy*ty, sw*tw, sh*th],...] 
    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs



#test utils
if __name__ == '__main__':
    all_imgs, classes_count, class_mapping = get_data()
    pprint.pprint(all_imgs[:3])
    pprint.pprint(classes_count)
    pprint.pprint(class_mapping)

