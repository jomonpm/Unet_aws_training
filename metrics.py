"""Losses and metrics for binary segmentation.

Both operate on probabilities in [0, 1] - the model ends in a sigmoid, so no
extra activation is applied here.
"""

import tensorflow as tf

SMOOTH = 1e-15


@tf.keras.utils.register_keras_serializable(package="unet")
def dice_coef(y_true, y_pred):
    """Soft Dice coefficient over the whole batch."""
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + SMOOTH) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + SMOOTH
    )


@tf.keras.utils.register_keras_serializable(package="unet")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="unet")
def iou(y_true, y_pred):
    """Soft IoU / Jaccard index.

    Written in pure TensorFlow rather than wrapping NumPy in a
    `tf.numpy_function`: the numpy version forces a device sync on every batch
    and cannot be traced into the graph.
    """
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)


# Handy when loading a saved model: model = load_model(path, custom_objects=CUSTOM_OBJECTS)
CUSTOM_OBJECTS = {"dice_coef": dice_coef, "dice_loss": dice_loss, "iou": iou}
