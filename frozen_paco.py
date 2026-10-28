#!/usr/bin/env python3
import argparse
from pathlib import Path

import predict_count

MODEL_PATH = Path(__file__).parent / 'models' / 'best.pt'


def get_device():
    try:
        import torch
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def main():
    parser = argparse.ArgumentParser(prog='frozen_paco', description='Подсчёт клеток водорослей на снимках')
    parser.add_argument('-i',  '--input',  required=True,             help='папка с изображениями')
    parser.add_argument('-o',  '--output', required=True,             help='папка для результатов')
    parser.add_argument('-ct', '--conf',   type=float, default=0.5,   help='порог уверенности (0-1)')
    args = parser.parse_args()

    predict_count.run_cell_counter(
        img_path=args.input,
        output_dir=args.output,
        model_path=str(MODEL_PATH),
        conf_th=args.conf,
        device=get_device(),
    )


if __name__ == '__main__':
    main()
