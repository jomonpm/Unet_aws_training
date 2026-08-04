# U-Net Semantic Segmentation (TensorFlow / Keras)

A U-Net built from scratch in Keras for binary segmentation, with Dice loss,
Dice/IoU metrics, and a training script set up to run either locally or on an
AWS GPU instance. Trained on a polygon-annotated indoor dataset.

```
train.py    dataset pipeline + training loop
unet.py     the architecture
metrics.py  Dice loss, Dice coefficient, IoU
predict.py  run a trained model over new images
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset layout

Images and masks are paired by sort order, so the *n*-th image must correspond
to the *n*-th mask. Masks are single-channel, white (255) for foreground:

```
data_root/
  train/
    images/*.jpg
    mask/*.png
  test/
    images/*.jpg
    mask/*.png
```

`load_dataset()` fails loudly if the counts don't match, which is the failure
mode that otherwise shows up as a model that trains but never converges.

## Training

```bash
# defaults: 512x512, batch 2, 20 epochs, lr 1e-4
python train.py --data-root /path/to/data_root

# smaller and faster, e.g. while checking the pipeline works
python train.py --data-root ./data --img-size 256 --batch-size 8 --epochs 40
```

Artefacts land in `files/`: the best checkpoint as `model.keras`, per-epoch
metrics as `history.csv`, and TensorBoard events under `files/logs`.

```bash
tensorboard --logdir files/logs
```

## Inference

```bash
python predict.py --model files/model.keras --input photo.jpg
python predict.py --model files/model.keras --input ./test_images --out-dir results
```

For each input it writes `<name>_mask.png` (binary, at the original resolution)
and `<name>_overlay.png` (prediction tinted green over the source image), and
prints what fraction of the frame was segmented.

## Design notes

**Loss.** Dice loss rather than binary cross-entropy, because the masks are
heavily imbalanced - the foreground is a small fraction of most frames, and BCE
lets the model score well by predicting "background" almost everywhere.

**Metrics.** Dice and IoU are computed in pure TensorFlow. The obvious way to
write IoU is to wrap NumPy in `tf.numpy_function`, but that forces a device sync
on every batch and can't be traced into the graph, so it is slower than the
model it's measuring on a GPU.

**Skip connections.** These are taken *after* each pooling step rather than
before it, so every skip arrives at half the decoder's resolution and is
bilinearly upsampled by `ResizeLayer` before concatenation. That differs from
the original paper, where skips carry full-resolution encoder features and are
cropped rather than resized. The trade-off is fewer parameters in the decoder
against less fine spatial detail reaching it.

**Serialisation.** `ResizeLayer` is a real Keras layer with `get_config()` and a
`register_keras_serializable` decorator, not an inline `tf.image.resize` call.
Inline ops work while the model is in memory and then fail to reload from
`.keras` - which you discover at inference time, after training has finished.

## Results

Training writes `files/history.csv`; fill in your best validation numbers here.

| | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| Validation | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

## Possible next steps

- Batch normalisation in the conv blocks, which the original U-Net predates
- Augmentation (flips, colour jitter) - currently there is none
- A held-out test split; `test/` is used for validation during training, so the
  reported numbers are not a clean generalisation estimate
