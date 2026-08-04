"""Run a trained U-Net over images and write out masks and overlays.

Usage:
    python predict.py --model files/model.keras --input photo.jpg
    python predict.py --model files/model.keras --input ./test_images --out-dir results
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf

from metrics import CUSTOM_OBJECTS
from unet import ResizeLayer

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def collect_images(source):
    if os.path.isfile(source):
        return [source]
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob(os.path.join(source, ext)))
    if not paths:
        raise FileNotFoundError(f"No images found in {source}")
    return sorted(paths)


def load_trained_model(model_path):
    custom = dict(CUSTOM_OBJECTS, ResizeLayer=ResizeLayer)
    return tf.keras.models.load_model(model_path, custom_objects=custom)


def predict_mask(model, image, size, threshold):
    """Return a uint8 mask at the model's resolution."""
    x = cv2.resize(image, (size, size)).astype("float32") / 255.0
    prob = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
    return ((prob[:, :, 0] > threshold) * 255).astype(np.uint8)


def overlay(image, mask, alpha=0.5):
    """Tint the predicted region green so you can eyeball the result."""
    coloured = np.zeros_like(image)
    coloured[:, :, 1] = mask
    return cv2.addWeighted(image, 1.0, coloured, alpha, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="files/model.keras")
    parser.add_argument("--input", required=True, help="image file or directory")
    parser.add_argument("--out-dir", default="predictions")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="probability cutoff for the mask"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_trained_model(args.model)
    paths = collect_images(args.input)
    print(f"Running on {len(paths)} image(s)")

    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skipped (unreadable): {path}")
            continue

        mask = predict_mask(model, image, args.img_size, args.threshold)
        # back to the original resolution so the mask lines up with the source
        mask_full = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.out_dir, f"{stem}_mask.png"), mask_full)
        cv2.imwrite(
            os.path.join(args.out_dir, f"{stem}_overlay.png"),
            overlay(image, mask_full),
        )
        coverage = float((mask_full > 0).mean())
        print(f"  {os.path.basename(path)}: {coverage:.1%} of pixels segmented")

    print(f"Wrote masks and overlays to {args.out_dir}/")


if __name__ == "__main__":
    main()
