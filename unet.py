"""U-Net architecture for binary segmentation.

Encoder blocks are two 3x3 convolutions followed by a 2x2 max pool; the decoder
mirrors that with transposed convolutions and concatenates the matching encoder
features. Skip connections are taken *after* the pooling step, so each one is
resized up to the decoder's resolution before concatenation (see ResizeLayer).
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


@tf.keras.utils.register_keras_serializable(package="unet")
class ResizeLayer(Layer):
    """Bilinear resize as a layer, so it survives model saving/loading.

    Doing `tf.image.resize` inline works while the model is in memory but turns
    into a bare TF op that Keras cannot serialise into `.keras` format.
    """

    def __init__(self, target_size, **kwargs):
        super().__init__(**kwargs)
        self.target_size = tuple(target_size)

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config


def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(inputs)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def maxpool(inputs):
    return tf.keras.layers.MaxPool2D((2, 2))(inputs)


def up_conv(inputs, num_filters):
    return tf.keras.layers.Conv2DTranspose(
        num_filters, (2, 2), strides=2, padding="same"
    )(inputs)


def concatenate(decoder_val, encoder_val):
    target_size = (decoder_val.shape[1], decoder_val.shape[2])
    encoder_val = ResizeLayer(target_size)(encoder_val)
    return tf.keras.layers.Concatenate()([decoder_val, encoder_val])


def unet(input_shape):
    """Build the model. `input_shape` is (H, W, 3); output is (H, W, 1)."""
    inputs = tf.keras.layers.Input(input_shape)

    enc1 = maxpool(conv_block(inputs, 64))
    enc2 = maxpool(conv_block(enc1, 128))
    enc3 = maxpool(conv_block(enc2, 256))
    enc4 = maxpool(conv_block(enc3, 512))

    bridge = conv_block(enc4, 1024)
    bridge = concatenate(up_conv(bridge, 1024), enc4)

    dec1 = concatenate(up_conv(conv_block(bridge, 512), 512), enc3)
    dec2 = concatenate(up_conv(conv_block(dec1, 256), 256), enc2)
    dec3 = concatenate(up_conv(conv_block(dec2, 128), 128), enc1)

    outputs = conv_block(dec3, 64)
    # sigmoid, because dice_loss/dice_coef expect probabilities in [0, 1]
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="U-Net")


if __name__ == "__main__":
    unet(input_shape=(512, 512, 3)).summary()
