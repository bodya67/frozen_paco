from sahi.slicing import slice_coco
import os
from yolo_tiler import YoloTiler, TileConfig

def run_sahi_tiler(image_dir, coco_json_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=coco_json_path,
        image_dir=image_dir,
        output_coco_annotation_file_name="sliced_annotations", # ИСПРАВЛЕНО ЗДЕСЬ
        output_dir=output_dir,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.1,  # Отсекает боксы, которые обрезаны больше чем на 90%
        out_ext=".jpg",
        verbose=True
    )

def run_yolo_tiler(inp_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    
    # Настраиваем параметры нарезки
    config = TileConfig(
        slice_wh=(640, 640),
        overlap_wh=(0.2, 0.2),
        include_negative_samples=True
    )
    
    # Запускаем нарезку
    tiler = YoloTiler(
        source=inp_dir,
        target=out_dir,
        config=config
    )
    
    tiler.run()



