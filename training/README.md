# Training

Scripts used to build `models/best.pt`. Not needed for counting.

Extra dependencies (on top of `requirements.txt`; use a CUDA build of PyTorch if you have a GPU):

```bash
pip install yolo-tiler matplotlib
```

## Data

YOLO format with two classes: `0` = `cell_1` (single cell), `1` = `cell_2` (doublet).

```
dataset/
├── data.yaml
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}
```

## Pipeline

```python
import slice_image, train

# 1. cut full images into 640x640 tiles with 20% overlap
slice_image.run_yolo_tiler('dataset', 'dataset_tiled')

# 2. train
train.train_yolo(model='yolo11n.pt', d_path='dataset_tiled', out_path='my_run',
                 n_epochs=100, image_size=640, batch_size=16, device=0)

# 3. count-based validation on full images across thresholds 0.1-0.9
#    -> count_validation.tsv + loss plots
train.validate_count_number(img_path='dataset/test/images', lbl_path='dataset/test/labels',
                            output_dir='val_counts',
                            model_path='runs/detect/my_run/weights/best.pt', device='cuda:0')
```

## Known issue: yolo_tiler crash

On some datasets tiling fails with:

```
AttributeError: 'Point' object has no attribute 'exterior'
```

A box that only touches a tile edge or corner produces a zero-area intersection, which `yolo_tiler` does not skip. Fix in `yolo_tiler/yolo_tiler.py`, right after `intersection = tile_polygon.intersection(box_polygon)`:

```python
if intersection.is_empty or intersection.area == 0:
    continue
```
