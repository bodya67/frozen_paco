from ultralytics import YOLO
import os


def train_yolo(model, d_path, out_path, n_epochs, image_size, batch_size, device):

    model = YOLO(model)
    
    model.train(
        data=d_path,
        epochs=n_epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        patience=30,
        name=out_path,
        workers=8,
        # project='test_run/detect',
        # --- augmentation tuned for round, orientation-invariant cells ---
        degrees=180.0,     # cells look the same at any rotation
        flipud=0.5,        # vertical flip is meaningful for a top-down view
        fliplr=0.5,
        translate=0.1,
        scale=0.3,
        mosaic=1.0,        # helps dense small-object detection
        close_mosaic=10,   # disable mosaic for the last N epochs to fine-tune
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        plots=True,
    )


