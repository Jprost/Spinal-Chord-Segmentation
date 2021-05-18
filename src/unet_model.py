from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanIoU
import tensorflow as tf

def conv_block(inputs: tf.tensor,
               filters: int,
               pool: bool=True) -> tf.tensor:
    """Build convolution block"""
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool:
        p = MaxPool2D((2, 2), padding='same')(x)
        return x, p
    else:
        return x


def build_unet(shape: tuple, num_classes: int) -> tf.tensor:
    """Build a simple U-net of 4 descending and ascending elements"""
    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True)
    x2, p2 = conv_block(p1, 32, pool=True)
    x3, p3 = conv_block(p2, 64, pool=True)
    x4, p4 = conv_block(p3, 128, pool=True)

    """ Bridge """
    b1 = conv_block(p4, 256, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)  # deconvolution
    c1 = Concatenate()([u1, x4])  # skip from the encoder shrinking
    x5 = conv_block(c1, 128, pool=False)  # Conv to augment feature depth

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 64, pool=False)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)

    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)

    return Model(inputs, output)


class Mean_IoU_custom(MeanIoU):
    """Creates a custom metric for training monitoring.
  Inherits from the tensorflow metric but applies a pre-processing step to the
  input.
  The prediction and labels are re-formated from a one hot-encoded label system
  to a multilabel system"""

    def __init__(self, num_classes: int, name: str="IoU"):
        # call the constructor of the parent class
        super(Mean_IoU_custom, self).__init__(num_classes, name=name)

    def update_state(self, y_true: tf.tensor,
                     y_pred: tf.tensor,
                     sample_weight: tf.tensor=None) -> None:
        # format the prediction and label to a mulitlabel plane
        y_true = tf.argmax(y_true, axis=3)
        y_pred = tf.argmax(y_pred, axis=3)
        # calls the IoU methods of the parent class
        return super().update_state(y_true, y_pred)
