import argparse
import random
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Argument parser                                                             #
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fine-tune YOLOv10 on a single YOLO-format dataset.",
    )
    parser.add_argument("--data_dir", required=True,
                        help="Folder containing images/ and labels/ sub-folders")
    parser.add_argument("--weights", default="./weights/pretrain_synCircle.pt",
                        help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (fixed)")
    parser.add_argument("--batch", type=int, default=5, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", default="0", help="'0', 'cpu', '0,1', etc.")
    parser.add_argument("--val_split", action="store_true",
                        help="Create 60/40 split and run validation")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    parser.add_argument("--names", nargs="+", default=["circle"],
                        help="Class names (index order)")
    return parser.parse_args()

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def make_split(root: Path):
    """Create train/val sub-folders (60/40) and return their relative paths."""
    img_dir, lbl_dir = root / "images", root / "labels"
    tr_img, val_img = img_dir / "train", img_dir / "val"
    tr_lbl, val_lbl = lbl_dir / "train", lbl_dir / "val"
    for d in (tr_img, val_img, tr_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_dir.glob("*") if p.is_file()]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * 0.6)
    train_imgs, val_imgs = imgs[:split_idx], imgs[split_idx:]

    for img_set, dst_img, dst_lbl in (
        (train_imgs, tr_img, tr_lbl),
        (val_imgs,   val_img, val_lbl),
    ):
        for img in img_set:
            lbl = lbl_dir / f"{img.stem}.txt"
            shutil.copy2(img, dst_img / img.name)
            shutil.copy2(lbl, dst_lbl / lbl.name)

    return tr_img.relative_to(root).as_posix(), val_img.relative_to(root).as_posix()

def write_yaml(root: Path, train_rel: str, val_rel: str, nc: int, names: list):
    """Write a temporary Ultralytics data YAML and return its path."""
    yml = {"path": str(root), "train": train_rel, "val": val_rel, "nc": nc, "names": names}
    yml_path = root / "temp_dataset.yaml"
    with open(yml_path, "w") as f:
        yaml.safe_dump(yml, f)
    return yml_path

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    root = Path(args.data_dir).expanduser().resolve()

    # 1. Determine training and validation directories
    if args.val_split:
        train_rel, val_rel = make_split(root)
        run_validation = True
    else:                                       
        train_rel = val_rel = "images"
        run_validation = False

    yaml_path = write_yaml(root, train_rel, val_rel, args.nc, args.names)

    # 2. Load model (no freezing)
    model = YOLO(args.weights)

    # 3. Train
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer="AdamW",
        lr0=1e-4,
        name=f"finetune_{root.name}",
        workers=4,
        val=run_validation,        # ← validation only if --val_split
        save_period=1,
    )

    print("\n✓ Training finished. Checkpoints are in runs/*/weights/.")

if __name__ == "__main__":
    main()