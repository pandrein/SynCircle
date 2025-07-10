import argparse, json
from pathlib import Path
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ───────────────────────────── Args ──────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Paths (with sensible defaults) ──
    p.add_argument("--weights", default="weights/pretrain_synCircle.pt",
                   help="Path to .pt checkpoint")
    p.add_argument("--source",  default="./datasets/demo/",
                   help="Folder of images OR dataset root")
    p.add_argument("--outdir",  default="results",
                   help="Where to save predictions (relative to dataset root or absolute path)")

    # ── Runtime opts ──
    p.add_argument("--device", default="0", help="GPU id or 'cpu'")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf",  type=float, default=0.25)

    # ── Saving (ON by default) ──
    p.add_argument("--save",    dest="save", action="store_true",
                   help="Save annotated images (ON by default)")
    p.add_argument("--no-save", dest="save", action="store_false",
                   help="Disable saving annotated images")
    p.set_defaults(save=True)
    p.add_argument("--save-txt", action="store_true",
                   help="Save YOLO‑format *.txt predictions alongside images")

    # ── Evaluation (ON by default) ──
    p.add_argument("--eval",    dest="eval", action="store_true",
                   help="Compute mAP/F1, needs labels")
    p.add_argument("--no-eval", dest="eval", action="store_false",
                   help="Skip evaluation entirely")
    p.set_defaults(eval=True)

    return p.parse_args()

# ─────────────────────────── Helpers ─────────────────────────────

def list_imgs(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT]

# ───────────────────────────── Main ──────────────────────────────

def main():
    args = parse_args()

    src_dir = Path(args.source).resolve()
    if not src_dir.is_dir():
        raise SystemExit(f"{src_dir} is not a directory.")

    # Resolve output directory
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate model
    model = YOLO(args.weights)

    # Decide which folder to feed to model.predict
    inf_dir = src_dir / "images" if args.eval else src_dir
    imgs = list_imgs(inf_dir)
    if not imgs:
        which = "images sub-folder" if args.eval else "folder"
        raise SystemExit(f"No images found in the {which}: {inf_dir}")

    # ── Build predict kwargs ──
    predict_kwargs = dict(
        source=str(inf_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        save_txt=args.save_txt,
        device=args.device,
        project=str(out_dir),
        name="",  
        exist_ok=True,
        stream=False,
        show=False,
    )

    # ────────────── Inference ──────────────
    model.predict(**predict_kwargs)

    # ────────────── Evaluation ─────────────
    if args.eval:
        (src_dir / "labels").mkdir(exist_ok=True)
        data_yaml = src_dir / "temp_eval.yaml"
        data_yaml.write_text(
            f"path: {src_dir}\n"
            f"train: images      # placeholder required by Ultralytics\n"
            f"val:   images\n"
            f"names:\n"
            f"  0: circle\n",
            encoding="utf-8",
        )

        val_metrics = model.val(
            data=str(data_yaml),
            batch=1,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
        )
        data_yaml.unlink(missing_ok=True)

        out = {
            "f1":      float(val_metrics.box.f1[0]),
            "map50":   float(val_metrics.box.map50),
            "map5095": float(val_metrics.box.map),
        }

        out_path = out_dir / "metrics.json"
        out_path.write_text(json.dumps(out, indent=4))
        print("\nValidation metrics:")
        for k, v in out.items():
            print(f"  {k:<8}: {v:.4f}")
        print(f"\nMetrics saved to {out_path}")

    print(f"Done ✔️. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
