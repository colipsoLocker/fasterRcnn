import tensorflow as tf
from tensorflow.keras.losses import   binary_crossentropy , categorical_crossentropy
from tensorflow.keras.backend import abs , sum , cast , less_equal , mean

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred   #rpn网络输出的是(0,width,heigh , 4*anchors)
        x_abs = abs(x)
        x_bool = cast(less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors): #rpn网络的类别损失函数
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * sum(y_true[:, :, :, :num_anchors] * binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / sum(epsilon + y_true[:, :, :, :num_anchors])
       
    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes): #最终的回归损失
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = abs(x)
		x_bool = cast(less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred): #最终的类别损失
	return lambda_cls_class * mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
