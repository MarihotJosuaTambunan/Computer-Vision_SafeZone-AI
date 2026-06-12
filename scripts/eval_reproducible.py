#!/usr/bin/env python3
"""
Reproducible evaluation script for SafeZone-AI

This script runs the detection model over a folder of images, saves per-image
predictions (JSON), annotated images, and a run manifest that contains
environment information (git commit/tag, package versions, seed, parameters).

Usage:
    python scripts/eval_reproducible.py --model ../weights/best_yolov10.onnx --data ./data/images --output ./eval_out

The script is deterministic by setting random seeds for Python, numpy and torch (if available).
"""

import argparse
import os
import sys
import time
import json
import random
import platform
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def gather_env_info():
    info = {}
    # Git info
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        tags = subprocess.check_output(['git', 'tag', '--points-at', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip().splitlines()
        info['git_commit'] = commit
        info['git_tags'] = tags
    except Exception:
        info['git_commit'] = None
        info['git_tags'] = []

    # Platform / python
    info['platform'] = platform.platform()
    info['python_version'] = platform.python_version()

    # Package versions (best effort)
    pkgs = ['ultralytics', 'numpy', 'opencv-python', 'torch', 'fastapi', 'streamlit']
    versions = {}
    try:
        # Python >=3.8
        import importlib.metadata as _m
    except Exception:
        try:
            import importlib_metadata as _m
        except Exception:
            _m = None

    for p in pkgs:
        try:
            if _m:
                versions[p] = _m.version(p)
            else:
                versions[p] = None
        except Exception:
            versions[p] = None

    info['package_versions'] = versions
    return info


def find_images(data_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    p = Path(data_dir)
    imgs = [str(x) for x in sorted(p.rglob('*')) if x.suffix.lower() in exts]
    return imgs


def annotate_and_save(image, predictions, out_path, class_names=None):
    img = image.copy()
    for det in predictions:
        x1, y1, x2, y2 = det['bbox']
        conf = det.get('confidence', 0.0)
        cid = det.get('class_id', 0)
        name = det.get('class_name') or (class_names.get(cid, str(cid)) if class_names else str(cid))
        color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(img, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), img)


def run_inference(model_path, images, conf_threshold=0.25, imgsz=640, device='cpu'):
    if YOLO is None:
        raise RuntimeError('ultralytics package not available. Install requirements.')

    model = YOLO(model_path)
    # try to obtain class mapping
    try:
        class_names = {int(k): v for k, v in getattr(model, 'names', {}).items()}
    except Exception:
        class_names = {}

    all_preds = {}
    stats = {'total_images': 0, 'total_detections': 0, 'per_class': {}}

    for img_path in images:
        stats['total_images'] += 1
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue

        res = model.predict(source=img, conf=conf_threshold, imgsz=imgsz, device=device, verbose=False)[0]

        preds = []
        if res.boxes is not None:
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                name = class_names.get(cls_id, str(cls_id))
                preds.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': name,
                })
                stats['total_detections'] += 1
                stats['per_class'][name] = stats['per_class'].get(name, 0) + 1

        all_preds[img_path] = preds

    return all_preds, stats, class_names


def main():
    parser = argparse.ArgumentParser(description='Reproducible evaluation runner')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (ONNX/pt)')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing images to run inference on')
    parser.add_argument('--output', type=str, default='./eval_out', help='Directory to store outputs')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for model')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference (cpu or cuda:0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'annotated_images').mkdir(exist_ok=True)

    print(f"Starting reproducible eval at {datetime.utcnow().isoformat()}Z")
    set_seed(args.seed)

    env = gather_env_info()
    env['run_timestamp'] = datetime.utcnow().isoformat() + 'Z'
    env['seed'] = args.seed
    env['model_path'] = args.model
    env['data_dir'] = args.data_dir
    env['conf_threshold'] = args.conf
    env['imgsz'] = args.imgsz
    env['device'] = args.device

    images = find_images(args.data_dir)
    if not images:
        print(f"No images found in {args.data_dir}")
        sys.exit(2)

    preds, stats, class_names = run_inference(args.model, images, conf_threshold=args.conf, imgsz=args.imgsz, device=args.device)

    # save predictions JSON
    preds_out = out_dir / 'predictions.json'
    with open(preds_out, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=2)

    # save annotated images
    for img_path, dets in preds.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        fname = Path(img_path).name
        out_img = out_dir / 'annotated_images' / fname
        annotate_and_save(img, dets, out_img, class_names=class_names)

    # save manifest
    manifest = {
        'env': env,
        'stats': stats,
        'num_images': len(images),
        'num_predictions': stats.get('total_detections', 0),
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Finished. Outputs written to: {out_dir.resolve()}")


if __name__ == '__main__':
    main()
