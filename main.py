# image_dir = "/home/bogdan/Загрузки/cell_counter.coco/train"
# coco_json_path = "/home/bogdan/Загрузки/cell_counter.coco/train/_annotations.coco.json"
# output_dir = "/home/bogdan/Загрузки/cell_counter.coco/test_out"



import slice_image
slice_image.run_yolo_tiler('/media/bogdan/FF57-FFD2/projects/cell_counter/data/yolov8', 
                           '/media/bogdan/FF57-FFD2/projects/cell_counter/data/yolov8_tiled')


import train

train.train_yolo(
    model='yolo11n.pt', 
    d_path='/media/bogdan/FF57-FFD2/projects/cell_counter/data/yolo_2class', 
    out_path='cell_count', 
    n_epochs=100, 
    image_size=640, 
    batch_size=16, 
    device=0
)

from ultralytics import YOLO
model = YOLO('runs/detect/cell_count/weights/best.pt')
val_results = model.val(
    data='/media/bogdan/FF57-FFD2/projects/cell_counter/data/yolo_2class',
    imgsz=640,
    batch=16,
    device=0
)

import predict_count

predict_count.run_cell_counter(img_path = '/media/bogdan/FF57-FFD2/projects/cell_counter/data/yolo_2class/valid/images', 
                               output_dir = "/media/bogdan/FF57-FFD2/projects/cell_counter/test", 
                               model_path = "/media/bogdan/FF57-FFD2/projects/cell_counter/runs/detect/cell_count-2/weights/best.pt", 
                               conf_th = 0.2, 
                               device = "cuda:0")