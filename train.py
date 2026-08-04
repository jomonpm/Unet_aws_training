"""Train the U-Net on a folder of images and binary masks.

Expected layout (mask filename must sort to the same position as its image):

    data_root/
      train/
        images/*.jpg
        mask/*.png
      test/
        images/*.jpg
        mask/*.png

Usage:
    python train.py --data-root /path/to/dataset
    python train.py --data-root ./data --img-size 256 --batch-size 4 --epochs 40
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

from metrics import dice_coef, dice_loss, iou
from unet import unet


def load_dataset(path):
    """Return sorted (images, masks) paths and check they line up."""
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "mask", "*.png")))

    if not images:
        raise FileNotFoundError(f"No .jpg images found under {path}/images")
    if len(images) != len(masks):
        raise ValueError(
            f"{path}: {len(images)} images but {len(masks)} masks - "
            "every image needs exactly one mask"
        )
    return images, masks


def read_image(image_path, size):
    image_path = image_path.decode()
    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if x is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    x = cv2.resize(x, (size, size))
    return (x.astype("float32")) / 255.0


def read_mask(mask_path, size):
    mask_path = mask_path.decode()
    x = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise ValueError(f"OpenCV could not read {mask_path}")
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_NEAREST)
    x = (x.astype("float32")) / 255.0
    return np.expand_dims(x, axis=-1)  # (size, size, 1)


def tf_parse(image_path, mask_path, size):
    def _parse(image_path, mask_path):
        return read_image(image_path, size), read_mask(mask_path, size)

    x, y = tf.numpy_function(_parse, (image_path, mask_path), (tf.float32, tf.float32))
    # numpy_function erases static shapes; Keras needs them back to build the graph
    x.set_shape([size, size, 3])
    y.set_shape([size, size, 1])
    return x, y


def tf_dataset(images, masks, batch_size, size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        # Keras can't shuffle a tf.data pipeline for us, and the file list comes
        # off disk sorted - without this the model sees the same batch order
        # every epoch.
        dataset = dataset.shuffle(len(images), reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda i, m: tf_parse(i, m, size), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


def steps_for(n, batch_size):
    return n // batch_size + (1 if n % batch_size else 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-root",
        required=True,
        help="dataset root containing train/ and test/ subfolders",
    )
    parser.add_argument("--img-size", type=int, default=512, help="square input size")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", default="files", help="where to write model + logs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.keras")

    train_x, train_y = load_dataset(os.path.join(args.data_root, "train"))
    valid_x, valid_y = load_dataset(os.path.join(args.data_root, "test"))
    print(f"train: {len(train_x)} images | valid: {len(valid_x)} images")

    train_dataset = tf_dataset(
        train_x, train_y, args.batch_size, args.img_size, shuffle=True
    )
    valid_dataset = tf_dataset(valid_x, valid_y, args.batch_size, args.img_size)

    model = unet((args.img_size, args.img_size, 3))
    model.compile(
        loss=dice_loss,
        optimizer=Adam(learning_rate=args.lr),
        metrics=[dice_coef, iou, Recall(), Precision()],
    )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1
        ),
        CSVLogger(os.path.join(args.out_dir, "history.csv")),
        TensorBoard(log_dir=os.path.join(args.out_dir, "logs")),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=valid_dataset,
        steps_per_epoch=steps_for(len(train_x), args.batch_size),
        validation_steps=steps_for(len(valid_x), args.batch_size),
        callbacks=callbacks,
    )
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
