from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from RoiPoolingConv import RoiPoolingConv
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file 
from tensorflow.keras.applications import VGG16
import numpy as np

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)    



def nn_base(input_tensor=None, trainable=False ,trainBySelf = False):


    # Determine proper input shape
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

 
    bn_axis = 3
    if trainBySelf:
    # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    else:
        base_model = VGG16(include_top=False,weights='imagenet')
        #block1
        x = base_model.get_layer('block1_conv1')(img_input)
        base_model.get_layer('block1_conv1').trainable = False
        x = base_model.get_layer('block1_conv2')(x)
        base_model.get_layer('block1_conv2').trainable = False
        x = base_model.get_layer('block1_pool')(x)
        base_model.get_layer('block1_pool').trainable = False
        #block2
        x = base_model.get_layer('block2_conv1')(x)
        base_model.get_layer('block2_conv1').trainable = False
        x = base_model.get_layer('block2_conv2')(x)
        base_model.get_layer('block2_conv2').trainable = False
        x = base_model.get_layer('block2_pool')(x)
        base_model.get_layer('block2_pool').trainable = False
        #block3
        x = base_model.get_layer('block3_conv1')(x)
        base_model.get_layer('block3_conv1').trainable = False
        x = base_model.get_layer('block3_conv2')(x)
        base_model.get_layer('block3_conv2').trainable = False
        x = base_model.get_layer('block3_conv3')(x)
        base_model.get_layer('block3_conv3').trainable = False
        x = base_model.get_layer('block3_pool')(x)
        base_model.get_layer('block3_pool').trainable = False
        #block4
        x = base_model.get_layer('block4_conv1')(x)
        base_model.get_layer('block4_conv1').trainable = False
        x = base_model.get_layer('block4_conv2')(x)
        base_model.get_layer('block4_conv2').trainable = False
        x = base_model.get_layer('block4_conv3')(x)
        base_model.get_layer('block4_conv3').trainable = False
        x = base_model.get_layer('block4_pool')(x)
        base_model.get_layer('block4_pool').trainable = False
        #block5
        x = base_model.get_layer('block5_conv1')(x)
        base_model.get_layer('block5_conv1').trainable = False
        x = base_model.get_layer('block5_conv2')(x)
        base_model.get_layer('block5_conv2').trainable = False
        x = base_model.get_layer('block5_conv3')(x)
        base_model.get_layer('block5_conv3').trainable = False

    return x

def rpn(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    #x_class 输出的是num_anchors 是否含有物体
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    #x_regr 输出的是对应的四个坐标
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    #print("Debug rpn layer output shape")
    #print(np.shape(x_class))
    #print(np.shape(x_regr))
    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    pooling_regions = 7
    input_shape = (num_rois,7,7,512)

    #(1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois]) #每个感兴趣的区域一层

    #通过TimeDistributed 把每个感兴趣的区域分别Flatten, Dense , dropout ,Dense来做分类，以及线性回归坐标位置。
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]



