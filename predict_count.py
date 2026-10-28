from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from collections import Counter
import cv2
import pandas as pd
import os

def run_cell_counter(img_path, output_dir, model_path, conf_th, device):
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,  # your best.pt
        confidence_threshold=conf_th,
        device=device,          # or "cpu"
    )
    
    # 2) read one image and run tiled prediction
    
    df_to_save = pd.DataFrame(columns=['img_name','cell_1', 'cell_2', 'total'])
    for img in os.listdir(img_path):
        if not img.lower().endswith((".jpg", ".png", ".tif")):
            continue
        print(f'On image: {img}')
        
        image = cv2.imread(img_path + '/' + img) 
        
        result = get_sliced_prediction(
            image[:, :, ::-1],                      # BGR -> RGB
            model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            # postprocess_type="NMS",
            # postprocess_match_metric="IOU",
            # postprocess_match_threshold=args.iou,
            # verbose=0,
        )
        
        result.export_visuals(export_dir=output_dir, file_name=img, hide_conf=True, text_size=1, rect_th=5)
        counts = Counter(op.category.name for op in result.object_prediction_list)
        df_to_save.loc[len(df_to_save)] = [img, counts['cell_1'], counts['cell_2'], counts['cell_1'] + (counts['cell_2'] * 2)]


    df_to_save.to_csv(output_dir + '/count_results.tsv', sep='\t', index=False, header=True)
