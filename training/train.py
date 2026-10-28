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
        # --- augmentation tuned for round, orientation-invariant cells ---
        degrees=180.0,     # cells look the same at any rotation
        flipud=0.5,        # vertical flip is meaningful for a top-down view
        fliplr=0.5,
        translate=0.1,
        scale=0.8,
        cls=1.5,            # weight class error. Set larger value if you need too be more precise on classes. 
        mosaic=1.0,        # helps dense small-object detection
        close_mosaic=10,   # disable mosaic for the last N epochs to fine-tune
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        plots=True,
    )


    
def validate_count_number(img_path, lbl_path, output_dir, model_path, device):
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from collections import Counter
    import cv2
    import pandas as pd
    import os
    
    # create outp dir and the results file 
    os.makedirs(output_dir, exist_ok=True)
    df_to_save = pd.DataFrame(columns=['img_name',
                                       'cell_1_pred',
                                       'cell_2_pred',
                                       'total_pred',
                                       'cell_1_true',
                                       'cell_2_true',
                                       'total_true',
                                       'count_1_loss_abs',
                                       'count_2_loss_abs',
                                       'total_loss_abs',
                                       'count_1_loss_sq',
                                       'count_2_loss_sq',
                                       'total_loss_sq',
                                       'count_1_loss_%',
                                       'count_2_loss_%',
                                       'total_loss_%',
                                       'threshold'])

    
    for conf_th in [i/10 for i in range(1, 10)]:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,  # your best.pt
            confidence_threshold=conf_th,
            device=device,          # or "cpu"
        )
        
    
        for img in os.listdir(img_path):
            if not img.lower().endswith((".jpg", ".png", ".tif")):
                continue
            print(f'### On image: {img} ###')
            
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
            
    
            counts = Counter(op.category.name for op in result.object_prediction_list)
            
            ### Read labels file and extract true count for the validation 
            df_true_counts = pd.read_csv(lbl_path + '/' + os.path.splitext(img)[0] + '.txt', sep=' ', header=None,
                             names=['cls', 'x', 'y', 'w', 'h'])
            counts_true = (df_true_counts['cls']
              .replace({0: 'cell_1', 1: 'cell_2'})
              .value_counts()
              .reindex(['cell_1', 'cell_2'], fill_value=0))
            
            print('### saving the results ###')
            df_to_save.loc[len(df_to_save)] = [img, counts['cell_1'], 
                                               counts['cell_2'], 
                                               counts['cell_1'] + (counts['cell_2'] * 2),
                                               counts_true['cell_1'],
                                               counts_true['cell_2'],
                                               counts_true['cell_1'] + (counts_true['cell_2'] * 2),
                                               abs(counts_true['cell_1'] - counts['cell_1']),
                                               abs(counts_true['cell_2'] - counts['cell_2']),
                                               abs((counts_true['cell_1'] + counts_true['cell_2']) - (counts['cell_1'] + counts['cell_2'])),
                                               (counts_true['cell_1'] - counts['cell_1']) ** 2,
                                               (counts_true['cell_2'] - counts['cell_2']) ** 2,
                                               ((counts_true['cell_1'] + counts_true['cell_2']) - (counts['cell_1'] + counts['cell_2'])) ** 2,
                                               abs(counts_true['cell_1'] - counts['cell_1']) / (counts_true['cell_1'] + counts['cell_1'] + 1e-8),
                                               abs(counts_true['cell_2'] - counts['cell_2']) / (counts_true['cell_2'] + counts['cell_2'] + 1e-8),
                                               abs((counts_true['cell_1'] + counts_true['cell_2']) - (counts['cell_1'] + counts['cell_2'])) / (counts_true['cell_1'] + counts_true['cell_2'] + counts['cell_1'] + counts['cell_2'] + 1e-8),
                                               conf_th
                                               
                                               ]

    print('### The results file has been saved ###')
    df_to_save.to_csv(output_dir + '/count_validation.tsv', sep='\t', index=False, header=True)
    
    ### create and save plots for the results. 
    df = df_to_save
    import matplotlib.pyplot as plt

    LOSS_ABS = ['count_1_loss_abs', 'count_2_loss_abs', 'total_loss_abs']
    LOSS_SQ  = ['count_1_loss_sq',  'count_2_loss_sq',  'total_loss_sq']
    LOSS_PCT = ['count_1_loss_%',   'count_2_loss_%',   'total_loss_%']
    LOSSES   = LOSS_ABS + LOSS_SQ + LOSS_PCT

    df = df.copy()
    df[LOSS_PCT] = df[LOSS_PCT] * 100          # your % cols are 0-1 fractions

    for th, grp in df.groupby('threshold'):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        for ax, col in zip(axes.flat, LOSSES):
            ax.hist(grp[col], bins=20, density=True, edgecolor='black')
            ax.set_title(col)
            ax.set_xlabel('loss'); ax.set_ylabel('density')
        fig.suptitle(f'threshold = {th}')
        fig.tight_layout()
        fig.savefig(f'{output_dir}/density_th_{th}.png', dpi=120)
        plt.close(fig)

    means = df.groupby('threshold')[LOSSES].mean()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    for col in LOSS_ABS:
        ax1.plot(means.index, means[col], marker='o', label=col)
    for col in LOSS_SQ:
        ax2.plot(means.index, means[col], marker='o', label=col)
    for col in LOSS_PCT:
        ax3.plot(means.index, means[col], marker='o', label=col)

    for ax, title in ((ax1, 'absolute error'), (ax2, 'squared error'), (ax3, '% error')):
        ax.set_xlabel('confidence threshold'); ax.set_ylabel('mean loss')
        ax.set_title(title); ax.legend(); ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/mean_loss_vs_threshold.png', dpi=120)
    plt.close(fig)

