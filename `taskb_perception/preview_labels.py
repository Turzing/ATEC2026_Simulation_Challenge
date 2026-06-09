"""
检查自动标注是否正确 — 把 YOLO 标签画在 PNG 上

用法（采完数据拷回自己电脑后）:
    conda activate perception
    cd taskb_perception

    # 默认检查 ../datasets/real
    python preview_labels.py

    # 指定数据集路径
    python preview_labels.py --data ../datasets/real

    # 每张都出预览图（默认最多 50 张，避免太多）
    python preview_labels.py --data ../datasets/real --max 200 --all

输出:
    <data>/preview/vis_000000.png  …  带彩色框 + 类别名
    终端打印统计：总张数、空标签、各类数量
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASS_NAMES, IMG_W, IMG_H

CLASS_COLORS_BGR = {
    0: (0, 255, 255),    # sugar_box — 黄
    1: (255, 0, 255),    # mustard_bottle — 紫
    2: (0, 255, 0),      # banana — 绿
}


def parse_yolo_label(line: str, img_w: int, img_h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cid = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return cid, x1, y1, x2, y2


def draw_labels_on_image(img_bgr, labels, img_w, img_h):
    out = img_bgr.copy()
    for line in labels:
        parsed = parse_yolo_label(line, img_w, img_h)
        if parsed is None:
            continue
        cid, x1, y1, x2, y2 = parsed
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
        color = CLASS_COLORS_BGR.get(cid, (255, 255, 255))
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{cid}:{name}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, tag, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return out


def find_image_label_pairs(data_root: Path, split: str = "train"):
    """split: train | val | all(legacy flat)"""
    if split in ("train", "val"):
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split
        if img_dir.is_dir():
            return img_dir, lbl_dir
        return None, None
    candidates = [
        (data_root / "images" / "train", data_root / "labels" / "train"),
        (data_root / "images", data_root / "labels"),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.is_dir():
            return img_dir, lbl_dir
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Preview YOLO auto-labels on collected images")
    parser.add_argument("--data", type=str, default="../datasets/real",
                        help="Dataset root (contains images/ and labels/)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"],
                        help="Preview train, val, or both splits")
    parser.add_argument("--out", type=str, default="",
                        help="Preview output dir (default: <data>/preview)")
    parser.add_argument("--max", type=int, default=50,
                        help="Max preview images to write")
    parser.add_argument("--all", action="store_true",
                        help="Preview every image (ignore --max)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Every N-th image (default 1)")
    parser.add_argument("--show", action="store_true",
                        help="Pop up windows (needs display)")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split_name in splits:
        _run_preview_split(data_root, split_name, args)


def _run_preview_split(data_root: Path, split_name: str, args):
    img_dir, lbl_dir = find_image_label_pairs(data_root, split_name)
    if img_dir is None:
        print(f"ERROR: no images/{split_name} under {data_root}")
        sys.exit(1)

    out_dir = Path(args.out).resolve() if args.out else (data_root / "preview" / split_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    if not images:
        print(f"ERROR: no PNG/JPG in {img_dir}")
        sys.exit(1)

    stats = {
        "total": len(images),
        "empty_label": 0,
        "with_label": 0,
        "class_counts": {0: 0, 1: 0, 2: 0},
        "total_boxes": 0,
    }

    preview_limit = len(images) if args.all else args.max
    preview_written = 0

    print("=" * 60)
    print(f"  YOLO Label Preview — {split_name}")
    print(f"  Data:    {data_root}")
    print(f"  Images:  {img_dir}  ({stats['total']} files)")
    print(f"  Labels:  {lbl_dir}")
    print(f"  Preview: {out_dir}")
    print("=" * 60)

    for i, img_path in enumerate(images):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        labels = []
        if lbl_path.is_file():
            text = lbl_path.read_text(encoding="utf-8").strip()
            if text:
                labels = [ln for ln in text.splitlines() if ln.strip()]

        if not labels:
            stats["empty_label"] += 1
        else:
            stats["with_label"] += 1
            for line in labels:
                p = parse_yolo_label(line, IMG_W, IMG_H)
                if p:
                    stats["class_counts"][p[0]] = stats["class_counts"].get(p[0], 0) + 1
                    stats["total_boxes"] += 1

        should_preview = (i % args.stride == 0) and (preview_written < preview_limit)
        if not should_preview:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        vis = draw_labels_on_image(img, labels, w, h)

        nbox = len(labels)
        header = f"{img_path.name}  |  {nbox} boxes"
        if nbox == 0:
            header += "  (EMPTY — no auto label)"
        cv2.putText(vis, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        out_path = out_dir / f"vis_{img_path.stem}.png"
        cv2.imwrite(str(out_path), vis)
        preview_written += 1

        if args.show:
            cv2.imshow("label preview", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
    if args.show:
        cv2.destroyAllWindows()

    print(f"\n--- Statistics ---")
    print(f"  Total images:     {stats['total']}")
    print(f"  With labels:      {stats['with_label']}")
    print(f"  Empty labels:     {stats['empty_label']}")
    print(f"  Total boxes:      {stats['total_boxes']}")
    for cid, name in CLASS_NAMES.items():
        print(f"    class {cid} {name}: {stats['class_counts'].get(cid, 0)} boxes")
    print(f"\n  Preview saved:    {preview_written} images -> {out_dir}")
    print("\n  Open preview folder and check:")
    print("    - Box should wrap the object")
    print("    - Color/class name should match (0=yellow sugar, 1=purple bottle, 2=green banana)")
    print("    - vis_* with EMPTY = RGB saved but no GT projection that frame")
    print("=" * 60)


if __name__ == "__main__":
    main()
