import cv2
import os
from glob import glob
import tensorflow as tf
from unet import unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from metrices import dice_loss, dice_coef, iou
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np
H = 512
W = 512

def load_dataset(path):
    image_path_x = sorted(glob(os.path.join(path, "images", "*.jpg")))
    mask_path_y = sorted(glob(os.path.join(path, "mask", "*.png")))
    return image_path_x, mask_path_y

def read_image(image_path): #read image
    image_path = image_path.decode()

    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x.astype("float32")

    x = x / 255
    return x

def read_mask(mask_path): #read mask
    mask_path = mask_path.decode()
    x = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 
    x = cv2.resize(x, (W, H))
    x = x.astype("float32")

    x = x / 255
    x = np.expand_dims(x, axis=-1)              ## (512, 512, 1)

    return x

def tf_parse(image_path, mask_path):
    def _parse(image_path, mask_path):
        x = read_image(image_path)
        y = read_mask(mask_path)
        return x,y
    x, y = tf.numpy_function(_parse, (image_path, mask_path), (tf.float32, tf.float32))
    #x.set_shape([H, W, 3])
    #y.set_shape([H, W, 1])
    return x, y

def tf_dataset(image_path, mask_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_path, mask_path))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset


    
if __name__ ==  "__main__":

    batch_size = 1
    lr = 1e-4

    num_epochs = 20
    model_path = "/home/ec2-user/model.h5"


    train_data_path = "/home/ec2-user/kitchen/polygon/train/"
    valid_data_path = "/home/ec2-user/kitchen/polygon/test/"
    train_x, train_y = load_dataset(train_data_path) #loading dataset path
    valid_x, valid_y = load_dataset(valid_data_path)
    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)

    train_step = len(train_x) // batch_size
    valid_step = len(valid_x) //batch_size

    if len(train_x) % batch_size != 0:
        train_step += 1
    if len(valid_x) % batch_size != 0:
        valid_step += 1

    model = unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(),EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False)
                ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_step,
        validation_steps=valid_step,
        callbacks=callbacks
    ) 

