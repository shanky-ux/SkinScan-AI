from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

METADATA_URL = "https://dataverse.harvard.edu/api/access/datafile/4338392"
ZIP_URL = "https://dataverse.harvard.edu/api/access/datafile/3172585"

METADATA_PATH = Path(r"C:\Users\stars\Downloads\SkinScan-AI-main\SkinScan-AI-main\backend\HAM10000_metadata.tab")
ZIP_PATH = Path(r"C:\Users\stars\Downloads\SkinScan-AI-main\SkinScan-AI-main\backend\HAM10000_images_part_1.zip")

DATASET_ROOT = Path(r"C:\Users\stars\Downloads\SkinScan-AI-main\SkinScan-AI-main\frontend\Dataset")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"

CLASS_MAP = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

IMAGES_PER_CLASS = 100
VAL_IMAGES_PER_CLASS = 20

TRAINING_SCRIPT = Path(r"C:\Users\stars\Downloads\SkinScan-AI-main\SkinScan-AI-main\train_skin_disease_model.py")
MODEL_OUTPUT = Path(r"C:\Users\stars\Downloads\SkinScan-AI-main\SkinScan-AI-main\attached_assets\skin_disease_model_1755756972916.pth")


def download_file(url: str, dest: Path, desc: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {desc} already exists at {dest} ({dest.stat().st_size} bytes)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {desc} from {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    tmp = dest.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)
    print(f"[done] {desc} saved to {dest} ({dest.stat().st_size} bytes)")


def parse_metadata(path: Path) -> dict[str, str]:
    print(f"[parse] Reading metadata from {path}")
    image_to_class: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            image_id = row.get("image_id", "").strip()
            dx = row.get("dx", "").strip()
            if image_id and dx:
                image_to_class[image_id] = dx
    print(f"[parse] Found {len(image_to_class)} image-class mappings")
    return image_to_class


def get_available_image_ids(zip_path: Path) -> set[str]:
    print(f"[inspect] Scanning zip contents: {zip_path}")
    available: set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            base = os.path.splitext(os.path.basename(name))[0]
            if base.startswith("ISIC_"):
                available.add(base)
            else:
                available.add(base)
    print(f"[inspect] Found {len(available)} image IDs in zip")
    return available


def create_dirs() -> None:
    for class_name in CLASS_MAP.values():
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)


def select_images_per_class(image_to_class: dict[str, str], available_ids: set[str], n_per_class: int) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {k: [] for k in CLASS_MAP}

    for image_id, dx in image_to_class.items():
        if dx in filtered and image_id in available_ids:
            filtered[dx].append(image_id)

    selected: dict[str, list[str]] = {k: [] for k in CLASS_MAP}
    rng = random.Random(42)
    for dx in CLASS_MAP:
        rng.shuffle(filtered[dx])
        selected[dx] = filtered[dx][:n_per_class]
        print(f"[select] {dx}: selected {len(selected[dx])} images")

    return selected


def extract_images(zip_path: Path, selected: dict[str, list[str]]) -> None:
    print(f"[extract] Extracting images from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        name_to_lower = {n: n.lower() for n in all_names}

        all_image_ids = [img_id for ids in selected.values() for img_id in ids]
        matches: dict[str, str] = {}

        for image_id in all_image_ids:
            candidates = [
                f"{image_id}.jpg",
                f"ISIC_{image_id}.jpg",
            ]
            found = None
            for cand in candidates:
                cl = cand.lower()
                for orig, lower in name_to_lower.items():
                    if lower == cl or lower.endswith("/" + cl):
                        if os.path.basename(orig).lower() == cl:
                            found = orig
                            break
                if found:
                    break
            if found:
                matches[image_id] = found
            else:
                print(f"[warn] Could not find zip member for image_id={image_id}")

        print(f"[extract] Found {len(matches)}/{len(all_image_ids)} matching members")

        for dx, img_ids in selected.items():
            class_name = CLASS_MAP[dx]
            dest_dir = TRAIN_DIR / class_name
            for image_id in img_ids:
                member_name = matches.get(image_id)
                if not member_name:
                    continue
                ext = os.path.splitext(member_name)[1]
                dest_path = dest_dir / f"{image_id}{ext}"
                with zf.open(member_name) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("[extract] Extraction complete")


def create_val_split() -> None:
    print(f"[split] Creating validation split with {VAL_IMAGES_PER_CLASS} images per class")
    for dx, class_name in CLASS_MAP.items():
        train_class_dir = TRAIN_DIR / class_name
        val_class_dir = VAL_DIR / class_name
        images = sorted([p for p in train_class_dir.iterdir() if p.is_file()])
        rng = random.Random(42)
        rng.shuffle(images)
        val_images = images[:VAL_IMAGES_PER_CLASS]
        for img in val_images:
            dst = val_class_dir / img.name
            shutil.move(str(img), str(dst))
        print(f"[split] {class_name}: moved {len(val_images)} to val")


def run_training() -> None:
    print("[train] Starting training")
    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--data-dir", str(DATASET_ROOT),
        "--output", str(MODEL_OUTPUT),
        "--architecture", "efficientnet_b0",
        "--epochs", "5",
        "--batch-size", "8",
        "--image-size", "224",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        print(f"[train] ERROR: training failed with return code {proc.returncode}")
        sys.exit(proc.returncode)
    print("[train] Training completed successfully")


def main() -> int:
    download_file(METADATA_URL, METADATA_PATH, "HAM10000 metadata")
    download_file(ZIP_URL, ZIP_PATH, "HAM10000 images part 1")

    image_to_class = parse_metadata(METADATA_PATH)
    available_ids = get_available_image_ids(ZIP_PATH)
    create_dirs()

    selected = select_images_per_class(image_to_class, available_ids, IMAGES_PER_CLASS)
    extract_images(ZIP_PATH, selected)

    create_val_split()

    run_training()

    print("\n=== Dataset Summary ===")
    for dx, class_name in CLASS_MAP.items():
        train_count = len(list((TRAIN_DIR / class_name).iterdir()))
        val_count = len(list((VAL_DIR / class_name).iterdir()))
        print(f"{class_name} ({dx}): train={train_count}, val={val_count}")

    if MODEL_OUTPUT.exists():
        print(f"\n[verify] Checkpoint exists at {MODEL_OUTPUT}")
        metadata_path = MODEL_OUTPUT.with_suffix(".json")
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            print(f"[verify] Best train metrics: {meta.get('best_train_metrics')}")
            print(f"[verify] Best val metrics: {meta.get('best_val_metrics')}")
            print(f"[verify] Architecture: {meta.get('architecture')}")
            print(f"[verify] Classes: {meta.get('class_names')}")
    else:
        print("[verify] ERROR: checkpoint not found")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
