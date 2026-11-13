"""
FedMedVision — Real Medical Dataset Downloader & Streaming 224x224 Resizer
Downloads:
  1. NIH ChestX-ray14 (nih-chest-xrays/data)
  2. CheXpert-v1.0-small (ashery/chexpert)
  3. VinDr-CXR Resized (awsaf49/vinbigdata-chest-xray-resized-png-256x256)
Selectively resizes to 224x224 RGB PNGs in data/processed/ and cleans raw archives to preserve disk.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import zipfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image

from utils.config_loader import load_config
from utils.logger import get_logger, log_system_event

logger = get_logger("fedmedvision.download")


def check_kaggle_auth() -> bool:
    """Check if Kaggle API credentials exist in environment or ~/.kaggle/."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    user_home = Path.home()
    if (user_home / ".kaggle" / "kaggle.json").is_file():
        return True
    if (user_home / ".kaggle" / "access_token").is_file():
        return True
    return False


def require_kaggle_auth() -> None:
    if not check_kaggle_auth():
        user_home = Path.home()
        msg = (
            "\n" + "=" * 70 + "\n"
            "❌ Kaggle API credentials not found!\n\n"
            "To download real clinical datasets from Kaggle:\n"
            "  1. Go to https://www.kaggle.com/settings -> 'Create New Token'\n"
            f"  2. Save the downloaded 'kaggle.json' to: {user_home / '.kaggle' / 'kaggle.json'}\n"
            "     OR set environment variables: KAGGLE_USERNAME and KAGGLE_KEY\n"
            "     OR set KAGGLE_API_TOKEN\n"
            "=" * 70 + "\n"
        )
        print(msg, flush=True)
        # Flush buffers and terminate cleanly without triggering repeated stdout callbacks
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(1)


def resize_image_to_224(src_path: Path, dest_path: Path, size: int = 224) -> bool:
    """Resize image to square size (default 224x224) in RGB format."""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            resized = rgb_img.resize((size, size), Image.Resampling.BILINEAR)
            resized.save(dest_path, format="PNG")
        return True
    except Exception as e:
        logger.warning(f"Error resizing {src_path}: {e}")
        return False


def download_and_process_nih(
    raw_dir: Path,
    processed_dir: Path,
    target_size: int = 224,
    cleanup_raw: bool = False,
    sample_limit: int = 10500,
) -> None:
    """Download NIH ChestX-ray14 dataset, parse BBox annotations and resize subset."""
    logger.info("=== Processing NIH ChestX-ray14 ===")
    require_kaggle_auth()
    import kaggle

    nih_raw = raw_dir / "nih_cxr14"
    nih_raw.mkdir(parents=True, exist_ok=True)
    nih_proc = processed_dir / "nih_cxr14"
    nih_proc.mkdir(parents=True, exist_ok=True)

    # Download dataset via kaggle API
    logger.info("Downloading NIH ChestX-ray14 dataset from Kaggle (nih-chest-xrays/data)...")
    kaggle.api.dataset_download_files("nih-chest-xrays/data", path=str(nih_raw), unzip=True)

    # Locate Data_Entry_2017.csv and BBox_List_2017.csv
    data_entry_csv = list(nih_raw.rglob("Data_Entry_2017.csv"))
    bbox_csv = list(nih_raw.rglob("BBox_List_2017.csv"))

    if not data_entry_csv:
        raise FileNotFoundError(f"Data_Entry_2017.csv not found in {nih_raw}")

    data_entry_path = data_entry_csv[0]
    bbox_path = bbox_csv[0] if bbox_csv else None

    # Copy metadata files into processed directory for reference
    shutil.copy2(data_entry_path, nih_proc / "Data_Entry_2017.csv")
    if bbox_path and bbox_path.is_file():
        shutil.copy2(bbox_path, nih_proc / "BBox_List_2017.csv")

    df_entry = pd.read_csv(data_entry_path)
    logger.info(f"NIH Data_Entry loaded: {len(df_entry)} entries.")

    bbox_image_indices = set()
    if bbox_path and bbox_path.is_file():
        df_bbox = pd.read_csv(bbox_path)
        bbox_image_indices = set(df_bbox["Image Index"].unique())
        logger.info(f"NIH BBox images detected: {len(bbox_image_indices)} unique images.")

    # Stratified selection: ensure all bbox images are selected, then fill up to sample_limit
    selected_indices = list(bbox_image_indices)
    remaining_needed = max(0, sample_limit - len(selected_indices))
    remaining_df = df_entry[~df_entry["Image Index"].isin(bbox_image_indices)]
    sampled_remaining = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=42)
    selected_indices.extend(sampled_remaining["Image Index"].tolist())

    logger.info(f"Resizing {len(selected_indices)} NIH images to {target_size}x{target_size}...")
    nih_img_dest = nih_proc / "images"
    nih_img_dest.mkdir(parents=True, exist_ok=True)

    for idx_name in selected_indices:
        found_imgs = list(nih_raw.rglob(idx_name))
        if found_imgs:
            resize_image_to_224(found_imgs[0], nih_img_dest / idx_name, size=target_size)

    if cleanup_raw:
        logger.info(f"Cleaning raw NIH directory: {nih_raw}")
        shutil.rmtree(nih_raw, ignore_errors=True)
    logger.info("NIH ChestX-ray14 processing complete.")


def download_and_process_chexpert(
    raw_dir: Path,
    processed_dir: Path,
    target_size: int = 224,
    cleanup_raw: bool = False,
    sample_limit: int = 10000,
) -> None:
    """Download CheXpert-v1.0-small dataset, parse labels and resize subset."""
    logger.info("=== Processing CheXpert ===")
    require_kaggle_auth()
    import kaggle

    chexpert_raw = raw_dir / "chexpert"
    chexpert_raw.mkdir(parents=True, exist_ok=True)
    chexpert_proc = processed_dir / "chexpert"
    chexpert_proc.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading CheXpert dataset from Kaggle (ashery/chexpert)...")
    kaggle.api.dataset_download_files("ashery/chexpert", path=str(chexpert_raw), unzip=True)

    train_csv = list(chexpert_raw.rglob("train.csv"))
    if not train_csv:
        raise FileNotFoundError(f"train.csv not found in {chexpert_raw}")

    train_csv_path = train_csv[0]
    shutil.copy2(train_csv_path, chexpert_proc / "train.csv")

    df_train = pd.read_csv(train_csv_path)
    logger.info(f"CheXpert train.csv loaded: {len(df_train)} entries.")

    sampled_df = df_train.sample(n=min(sample_limit, len(df_train)), random_state=42)
    chexpert_img_dest = chexpert_proc / "images"
    chexpert_img_dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resizing {len(sampled_df)} CheXpert images to {target_size}x{target_size}...")
    for _, row in sampled_df.iterrows():
        rel_p = row["Path"]
        filename = Path(rel_p).name
        patient_folder = Path(rel_p).parent.name
        dest_filename = f"{patient_folder}_{filename}"

        found_imgs = list(chexpert_raw.rglob(filename))
        if found_imgs:
            resize_image_to_224(found_imgs[0], chexpert_img_dest / dest_filename, size=target_size)

    if cleanup_raw:
        logger.info(f"Cleaning raw CheXpert directory: {chexpert_raw}")
        shutil.rmtree(chexpert_raw, ignore_errors=True)
    logger.info("CheXpert processing complete.")


def download_and_process_vindr(
    raw_dir: Path,
    processed_dir: Path,
    target_size: int = 224,
    cleanup_raw: bool = False,
    sample_limit: int = 1500,
) -> None:
    """Download VinDr-CXR resized dataset, parse labels and resize subset."""
    logger.info("=== Processing VinDr-CXR ===")
    require_kaggle_auth()
    import kaggle

    vindr_raw = raw_dir / "vindr_cxr"
    vindr_raw.mkdir(parents=True, exist_ok=True)
    vindr_proc = processed_dir / "vindr_cxr"
    vindr_proc.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading VinDr-CXR dataset from Kaggle (awsaf49/vinbigdata-chest-xray-resized-png-256x256)...")
    kaggle.api.dataset_download_files("awsaf49/vinbigdata-chest-xray-resized-png-256x256", path=str(vindr_raw), unzip=True)

    train_csv = list(vindr_raw.rglob("train.csv"))
    if not train_csv:
        raise FileNotFoundError(f"train.csv not found in {vindr_raw}")

    train_csv_path = train_csv[0]
    shutil.copy2(train_csv_path, vindr_proc / "train.csv")

    df_train = pd.read_csv(train_csv_path)
    logger.info(f"VinDr train.csv loaded: {len(df_train)} entries.")

    unique_img_ids = df_train["image_id"].unique()
    rng_df = pd.Series(unique_img_ids).sample(n=min(sample_limit, len(unique_img_ids)), random_state=42)
    selected_ids = set(rng_df.tolist())

    vindr_img_dest = vindr_proc / "images"
    vindr_img_dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Resizing {len(selected_ids)} VinDr images to {target_size}x{target_size}...")
    for img_id in selected_ids:
        filename = f"{img_id}.png"
        found_imgs = list(vindr_raw.rglob(filename))
        if found_imgs:
            resize_image_to_224(found_imgs[0], vindr_img_dest / filename, size=target_size)

    if cleanup_raw:
        logger.info(f"Cleaning raw VinDr directory: {vindr_raw}")
        shutil.rmtree(vindr_raw, ignore_errors=True)
    logger.info("VinDr-CXR processing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMedVision Real Clinical Dataset Acquisition & Resizer")
    parser.add_argument(
        "--dataset",
        choices=["all", "nih", "chexpert", "vindr"],
        default="all",
        help="Which clinical dataset to download and process",
    )
    parser.add_argument("--resize", type=int, default=224, help="Target image resolution (default: 224)")
    parser.add_argument("--cleanup-raw", action="store_true", help="Delete multi-GB raw downloads after resizing")
    args = parser.parse_args()

    config = load_config()
    raw_dir = config.project_root / "data" / "raw"
    processed_dir = config.project_root / "data" / "processed"

    log_system_event(logger, "dataset_download_start", dataset=args.dataset, resize=args.resize)

    if args.dataset in ("all", "nih"):
        download_and_process_nih(raw_dir, processed_dir, target_size=args.resize, cleanup_raw=args.cleanup_raw)

    if args.dataset in ("all", "chexpert"):
        download_and_process_chexpert(raw_dir, processed_dir, target_size=args.resize, cleanup_raw=args.cleanup_raw)

    if args.dataset in ("all", "vindr"):
        download_and_process_vindr(raw_dir, processed_dir, target_size=args.resize, cleanup_raw=args.cleanup_raw)

    log_system_event(logger, "dataset_download_complete")
    print("✅ Clinical dataset acquisition and processing completed.")


if __name__ == "__main__":
    main()
