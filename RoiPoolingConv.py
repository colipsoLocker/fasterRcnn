from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import shape , cast ,reshape ,concatenate , permute_dimensions
from tensorflow.image import resize_images

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, pool_size, pool_size ，channels)`

    
    输入的是个4维的tensor， [X_img,X_roi] ，每个又都是个三维图片(1, rows, cols, channels)
    '''
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        #x是输入的张量，call函数就是计算从输入到输出。
        #输入的是两个网络的输出，一个是basenet，另一个是input_roi 是输入的感兴趣的区域。shape=(None, 4)

        assert(len(x) == 2)

        img = x[0] #(1, rows, cols, channels)
        rois = x[1] #(1,num_rois,4)   with ordering (x,y,w,h)
        #>>> K.shape(input).eval(session=tf_session)
        #array([2, 4, 5], dtype=int32)
        input_shape = shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):#num_rois = 4 一次感兴趣的区域个数

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]
            
            row_length = w / float(self.pool_size)# self.pool_size 输出的固定的方阵 exp：7X7 为了输出剪裁的单位
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            #NOTE: the RoiPooling implementation differs between theano and tensorflow due to the lack of a resize op
            # in theano. The theano implementation is much less efficient and leads to long compile times
            """Casts a tensor to a different dtype and returns it.

            You can cast a Keras variable but it still returns a Keras tensor.

            Arguments:
                x: Keras tensor (or variable).
                dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

            Returns:
                Keras tensor with dtype `dtype`.

            Example:
            ```python
                >>> from keras import backend as K
                >>> input = K.placeholder((2, 3), dtype='float32')
                >>> input
                <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
                # It doesn't work in-place as below.
                >>> K.cast(input, dtype='float16')
                <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
                >>> input
                <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
                # you need to assign it.
                >>> input = K.cast(input, dtype='float16')
                >>> input
                <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
            ```
            """
            x = cast(x, 'int32')
            y = cast(y, 'int32')
            w = cast(w, 'int32')
            h = cast(h, 'int32')

            #把感兴趣的区域ROI 都改尺寸到poolSize * poolSize
            rs = resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = concatenate(outputs, axis=0)
        final_output = reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = permute_dimensions(final_output, (0, 1, 2, 3, 4)) #不用改变tensor的纬度

        return final_output  #[每个感兴趣的区域ROIpool后的结果，](1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
