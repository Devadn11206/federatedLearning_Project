"""
FedMedVision — Cross-Hospital Data Consortium Preparation Script
Supports real clinical chest radiograph archives (NIH ChestX-ray14, CheXpert, VinDr-CXR)
and an instant `--synthetic-dryrun` mode generating synthetic chest phantom scans.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from utils.config_loader import load_config
from utils.logger import get_logger, log_system_event
from utils.reproducibility import seed_everything

logger = get_logger("fedmedvision.data")


def generate_chest_phantom(
    path: Path,
    seed: int,
    image_size: int = 224,
    has_effusion: bool = False,
    has_cardiomegaly: bool = False,
    has_infiltration: bool = False,
    has_pneumonia: bool = False,
) -> None:
    """
    Generate a lightweight synthetic 224x224 grayscale/RGB chest phantom image
    with simulated anatomical landmarks (ribs, lung fields, cardiac silhouette)
    and condition-specific artifacts.
    """
    rng = np.random.RandomState(seed)
    img = Image.new("L", (image_size, image_size), color=30)
    draw = ImageDraw.Draw(img)

    # Torso contour
    draw.ellipse([20, 10, image_size - 20, image_size + 40], fill=60)

    # Lung fields (left & right lobes)
    draw.ellipse([35, 40, 95, 180], fill=15)
    draw.ellipse([image_size - 95, 40, image_size - 35, 180], fill=15)

    # Spine/mediastinum center column
    draw.rectangle([100, 30, 124, 210], fill=120)

    # Rib cage shadows
    for y in range(50, 180, 20):
        draw.arc([30, y - 10, 105, y + 15], start=180, end=360, fill=75, width=2)
        draw.arc([image_size - 105, y - 10, image_size - 30, y + 15], start=180, end=360, fill=75, width=2)

    # Cardiac silhouette (Heart in left lower hemithorax)
    heart_w = 45 if has_cardiomegaly else 30
    draw.ellipse([95 - heart_w // 2, 120, 120 + heart_w, 185], fill=140)

    # Pathology artifacts
    if has_effusion:
        draw.polygon([(35, 180), (95, 180), (95, 155), (35, 170)], fill=110)
        draw.polygon([(image_size - 95, 180), (image_size - 35, 180), (image_size - 35, 170), (image_size - 95, 155)], fill=110)

    if has_pneumonia or has_infiltration:
        patch_x = rng.randint(45, 80)
        patch_y = rng.randint(60, 130)
        draw.ellipse([patch_x, patch_y, patch_x + 30, patch_y + 25], fill=130)

    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=4.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    final_img = Image.fromarray(arr).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(path, format="PNG")


def create_dirichlet_skew(
    num_samples: int,
    pathologies: list[str],
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Generate multi-label binary vectors skewed by a Dirichlet prior."""
    rng = np.random.RandomState(seed)
    num_classes = len(pathologies)
    class_priors = rng.dirichlet(alpha=np.full(num_classes, alpha))
    target_rates = np.clip(class_priors * 1.5, 0.05, 0.45)

    labels = np.zeros((num_samples, num_classes), dtype=int)
    for c in range(num_classes):
        if pathologies[c] == "No Finding":
            continue
        labels[:, c] = (rng.rand(num_samples) < target_rates[c]).astype(int)

    disease_mask = labels[:, :num_classes].sum(axis=1) > 0
    if "No Finding" in pathologies:
        nf_idx = pathologies.index("No Finding")
        labels[~disease_mask, nf_idx] = 1
        labels[disease_mask, nf_idx] = 0

    return labels


def partition_dataframe_dirichlet(
    df: pd.DataFrame,
    pathologies: list[str],
    num_partitions: int,
    alpha: float,
    seed: int,
) -> list[pd.DataFrame]:
    """
    Partition real labeled or unlabeled dataframe across nodes using Dirichlet sampling on primary finding.
    Ensures non-IID class skew across consortium nodes.
    """
    rng = np.random.RandomState(seed)
    partitions = [[] for _ in range(num_partitions)]

    def get_primary_class(row: pd.Series) -> str:
        for p in pathologies:
            if p != "No Finding" and row.get(p, 0) == 1:
                return p
        return "No Finding"

    df_copy = df.copy()
    df_copy["_primary"] = df_copy.apply(get_primary_class, axis=1)

    for p in pathologies:
        p_df = df_copy[df_copy["_primary"] == p]
        n_p = len(p_df)
        if n_p == 0:
            continue

        proportions = rng.dirichlet(np.full(num_partitions, alpha))
        counts = (proportions * n_p).astype(int)
        diff = n_p - counts.sum()
        counts[0] += diff

        indices = rng.permutation(p_df.index)
        cur = 0
        for i in range(num_partitions):
            cnt = counts[i]
            if cnt > 0:
                part_idx = indices[cur : cur + cnt]
                partitions[i].append(df.loc[part_idx])
                cur += cnt

    result_dfs = []
    for i in range(num_partitions):
        if partitions[i]:
            part_df = pd.concat(partitions[i], ignore_index=True)
            result_dfs.append(part_df.sample(frac=1.0, random_state=seed + i).reset_index(drop=True))
        else:
            result_dfs.append(pd.DataFrame(columns=df.columns))

    return result_dfs


def prepare_real_consortium(
    data_processed_dir: Path,
    splits_dir: Path,
    config: Any,
) -> None:
    """
    Parse processed real NIH ChestX-ray14, CheXpert, and VinDr-CXR datasets:
      - NIH ChestX-ray14: Parse Data_Entry_2017.csv pipe-separated Finding Labels & BBox_List_2017.csv
      - CheXpert: Map Cardiomegaly, Pleural Effusion -> Effusion, Lung Opacity -> Infiltration, Pneumonia, No Finding
      - VinDr-CXR: Multi-label zero-shot cohort
      - Partitions Hubs 0 & 1 with Dirichlet alpha=0.3
      - Partitions Clinics 2, 3, 4, 5 with is_labeled=False
      - Extracts clean val_test.csv, external_test.csv, and bbox_eval.csv
    """
    pathologies = config.consortium.pathologies
    alpha = config.consortium.dirichlet_alpha
    base_seed = config.consortium.seeds[0] if config.consortium.seeds else 42

    nih_proc_dir = data_processed_dir / "nih_cxr14"
    chexpert_proc_dir = data_processed_dir / "chexpert"
    vindr_proc_dir = data_processed_dir / "vindr_cxr"

    nih_entry_csv = nih_proc_dir / "Data_Entry_2017.csv"
    chexpert_train_csv = chexpert_proc_dir / "train.csv"

    if not nih_entry_csv.exists():
        raw_nih = config.project_root / "data" / "raw" / "nih_cxr14" / "Data_Entry_2017.csv"
        if raw_nih.exists():
            nih_entry_csv = raw_nih
        else:
            raise FileNotFoundError(f"NIH metadata not found at {nih_entry_csv}")

    logger.info("=== Processing Real Clinical NIH ChestX-ray14 Archive ===")
    df_nih = pd.read_csv(nih_entry_csv)
    nih_img_dir = nih_proc_dir / "images"
    existing_nih_images = set(p.name for p in nih_img_dir.glob("*.png")) if nih_img_dir.exists() else set()

    if existing_nih_images:
        df_nih = df_nih[df_nih["Image Index"].isin(existing_nih_images)].copy()
        logger.info(f"Matched {len(df_nih)} real NIH images on disk in {nih_img_dir}")
    else:
        logger.warning(f"No resized NIH PNGs found in {nih_img_dir}. Using metadata entries.")

    nih_records = []
    for _, row in df_nih.iterrows():
        img_name = row["Image Index"]
        findings = str(row["Finding Labels"]).split("|")
        rec = {
            "sample_id": f"nih_{Path(img_name).stem}",
            "image_path": f"nih_cxr14/images/{img_name}",
            "dataset": "nih_cxr14",
            "is_labeled": True,
        }
        for p in pathologies:
            if p == "No Finding":
                rec[p] = 1 if "No Finding" in findings else 0
            elif p == "Infiltration":
                rec[p] = 1 if ("Infiltration" in findings or "Infiltrate" in findings) else 0
            else:
                rec[p] = 1 if p in findings else 0
        nih_records.append(rec)

    df_nih_processed = pd.DataFrame(nih_records)
    logger.info(f"Loaded {len(df_nih_processed)} real labeled NIH CXR entries.")

    # Process BBox_List_2017.csv
    nih_bbox_csv = nih_proc_dir / "BBox_List_2017.csv"
    bbox_records = []
    if nih_bbox_csv.exists():
        df_raw_bbox = pd.read_csv(nih_bbox_csv)
        for _, b_row in df_raw_bbox.iterrows():
            img_idx = b_row["Image Index"]
            if existing_nih_images and img_idx not in existing_nih_images:
                continue
            finding = b_row["Finding Label"]
            finding_mapped = None
            if finding == "Infiltrate":
                finding_mapped = "Infiltration"
            elif finding in pathologies:
                finding_mapped = finding

            if finding_mapped:
                scale_factor = 224.0 / 1024.0
                raw_x = float(b_row.get("Bbox [x", b_row.get("x", 0.0)))
                raw_y = float(b_row.get("y", 0.0))
                raw_w = float(b_row.get("w", 0.0))
                raw_h = float(b_row.get("h]", b_row.get("h", 0.0)))

                b_rec = {
                    "sample_id": f"bbox_{Path(img_idx).stem}",
                    "image_path": f"nih_cxr14/images/{img_idx}",
                    "finding": finding_mapped,
                    "bbox_x": round(raw_x * scale_factor, 2),
                    "bbox_y": round(raw_y * scale_factor, 2),
                    "bbox_w": round(raw_w * scale_factor, 2),
                    "bbox_h": round(raw_h * scale_factor, 2),
                    "image_size": 224,
                }
                for p in pathologies:
                    b_rec[p] = 1 if p == finding_mapped else 0
                bbox_records.append(b_rec)

    bbox_df = pd.DataFrame(bbox_records)
    bbox_file = splits_dir / "bbox_eval.csv"
    bbox_df.to_csv(bbox_file, index=False)
    logger.info(f"Saved real BBox evaluation split to {bbox_file} ({len(bbox_df)} entries).")

    # Process CheXpert Dataset
    logger.info("=== Processing Real Clinical CheXpert Archive ===")
    chexpert_img_dir = chexpert_proc_dir / "images"
    existing_chex_images = set(p.name for p in chexpert_img_dir.glob("*.png")) if chexpert_img_dir.exists() else set()

    chex_records = []
    if chexpert_train_csv.exists():
        df_chex = pd.read_csv(chexpert_train_csv)
        for idx, row in df_chex.iterrows():
            rel_p = str(row["Path"])
            parts = rel_p.split("/")
            if len(parts) >= 5:
                patient = parts[2]
                study = parts[3]
                view = parts[4].replace(".jpg", ".png")
                dest_filename = f"{patient}_{study}_{view}"
            else:
                dest_filename = Path(rel_p).name.replace(".jpg", ".png")

            if dest_filename not in existing_chex_images:
                continue

            img_path = f"chexpert/images/{dest_filename}"
            rec = {
                "sample_id": f"chex_{idx:06d}",
                "image_path": img_path,
                "dataset": "chexpert",
                "is_labeled": False,
            }

            def map_chex_val(v: Any) -> int:
                try:
                    val = float(v)
                    return 1 if val == 1.0 else 0
                except Exception:
                    return 0

            rec["Cardiomegaly"] = map_chex_val(row.get("Cardiomegaly", 0))
            rec["Effusion"] = map_chex_val(row.get("Pleural Effusion", 0))
            rec["Infiltration"] = map_chex_val(row.get("Lung Opacity", 0))
            rec["Pneumonia"] = map_chex_val(row.get("Pneumonia", 0))
            rec["No Finding"] = map_chex_val(row.get("No Finding", 0))

            chex_records.append(rec)
            if len(chex_records) >= 10000:
                break

    df_chex_processed = pd.DataFrame(chex_records)
    logger.info(f"Loaded {len(df_chex_processed)} real CheXpert records from images on disk.")

    # Partition Labeled Urban Hubs (0 & 1)
    val_sample_n = min(1500, max(3, int(len(df_nih_processed) * 0.15)))
    val_df = df_nih_processed.sample(n=val_sample_n, random_state=base_seed)
    hubs_pool_df = df_nih_processed.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    val_file = splits_dir / "val_test.csv"
    val_df.to_csv(val_file, index=False)
    logger.info(f"Saved real Consortium Validation/Test split to {val_file} ({len(val_df)} entries).")

    hub_partitions = partition_dataframe_dirichlet(
        hubs_pool_df,
        pathologies=pathologies,
        num_partitions=2,
        alpha=alpha,
        seed=base_seed + 101,
    )
    hub_0_df = hub_partitions[0]
    hub_1_df = hub_partitions[1]

    # Ensure both hubs receive positive counts even on small cohorts
    if len(hub_0_df) == 0 or len(hub_1_df) == 0:
        half_n = max(1, len(hubs_pool_df) // 2)
        hub_0_df = hubs_pool_df.iloc[:half_n].copy().reset_index(drop=True)
        hub_1_df = hubs_pool_df.iloc[half_n:].copy().reset_index(drop=True)

    hub_0_df.to_csv(splits_dir / "hub_0.csv", index=False)
    hub_1_df.to_csv(splits_dir / "hub_1.csv", index=False)
    logger.info(f"Saved hub_0.csv ({len(hub_0_df)}) and hub_1.csv ({len(hub_1_df)}) with Dirichlet alpha={alpha}.")

    # Centralized pooled train split (Hub 0 + Hub 1)
    centralized_pooled_df = pd.concat([hub_0_df, hub_1_df], ignore_index=True)
    centralized_pooled_df.to_csv(splits_dir / "centralized_pooled_train.csv", index=False)
    logger.info(f"Saved centralized_pooled_train.csv ({len(centralized_pooled_df)} entries).")

    # Partition Unlabeled Clinics (2, 3, 4, 5)
    clinic_partitions = partition_dataframe_dirichlet(
        df_chex_processed,
        pathologies=pathologies,
        num_partitions=4,
        alpha=0.5,
        seed=base_seed + 202,
    )
    for c_idx, node_name in enumerate(["clinic_2", "clinic_3", "clinic_4", "clinic_5"]):
        c_df = clinic_partitions[c_idx]
        c_df.to_csv(splits_dir / f"{node_name}.csv", index=False)
        logger.info(f"Saved {node_name}.csv ({len(c_df)} samples).")

    # External Test Set (VinDr-CXR)
    ext_file = splits_dir / "external_test.csv"
    vindr_img_dir = config.project_root / "data" / "processed" / "vindr_cxr" / "images"
    existing_vindr_images = sorted(list(vindr_img_dir.glob("*.png"))) if vindr_img_dir.exists() else []

    ext_records = []
    if existing_vindr_images:
        for idx, img_p in enumerate(existing_vindr_images[:1500]):
            ext_records.append({
                "sample_id": f"vindr_{img_p.stem}",
                "image_path": f"vindr_cxr/images/{img_p.name}",
                "dataset": "vindr_cxr",
                "is_labeled": False,
                "Cardiomegaly": 0,
                "Effusion": 0,
                "Infiltration": 0,
                "Pneumonia": 0,
                "No Finding": 1,
            })
        ext_df = pd.DataFrame(ext_records)
    else:
        ext_df = df_nih_processed.sample(n=min(len(df_nih_processed), 15), random_state=base_seed + 888)

    ext_df.to_csv(ext_file, index=False)
    logger.info(f"Saved external_test.csv to {ext_file} ({len(ext_df)} samples).")

    # Save consortium distribution summary JSON
    stats: dict[str, Any] = {
        "mode": "real",
        "pathologies": pathologies,
        "nodes": {
            "node_0": {
                "name": config.consortium.nodes["node_0"]["name"],
                "role": "labeled_hub",
                "sample_count": len(hub_0_df),
                "class_distribution": {p: int(hub_0_df[p].sum()) for p in pathologies},
            },
            "node_1": {
                "name": config.consortium.nodes["node_1"]["name"],
                "role": "labeled_hub",
                "sample_count": len(hub_1_df),
                "class_distribution": {p: int(hub_1_df[p].sum()) for p in pathologies},
            },
            "node_2": {
                "name": config.consortium.nodes["node_2"]["name"],
                "role": "unlabeled_clinic",
                "sample_count": len(clinic_partitions[0]),
                "class_distribution": {p: int(clinic_partitions[0][p].sum()) for p in pathologies},
            },
            "node_3": {
                "name": config.consortium.nodes["node_3"]["name"],
                "role": "unlabeled_clinic",
                "sample_count": len(clinic_partitions[1]),
                "class_distribution": {p: int(clinic_partitions[1][p].sum()) for p in pathologies},
            },
            "node_4": {
                "name": config.consortium.nodes["node_4"]["name"],
                "role": "unlabeled_clinic",
                "sample_count": len(clinic_partitions[2]),
                "class_distribution": {p: int(clinic_partitions[2][p].sum()) for p in pathologies},
            },
            "node_5": {
                "name": config.consortium.nodes["node_5"]["name"],
                "role": "unlabeled_clinic",
                "sample_count": len(clinic_partitions[3]),
                "class_distribution": {p: int(clinic_partitions[3][p].sum()) for p in pathologies},
            },
        },
    }

    stats_file = splits_dir / "consortium_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved consortium distribution stats to {stats_file}")


def build_synthetic_consortium(
    output_dir: Path,
    splits_dir: Path,
    config: Any,
    num_hub_samples: int = 100,
    num_clinic_samples: int = 80,
    num_val_samples: int = 60,
    num_ext_samples: int = 50,
    num_bbox_samples: int = 30,
) -> None:
    """
    Construct complete synthetic dataset splits, images, and metadata for fast dry-run testing.
    """
    pathologies = config.consortium.pathologies
    alpha = config.consortium.dirichlet_alpha
    seeds = config.consortium.seeds
    base_seed = seeds[0] if seeds else 42

    synthetic_img_dir = output_dir / "synthetic"
    synthetic_img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"mode": "synthetic", "pathologies": pathologies, "nodes": {}}

    logger.info("Generating synthetic consortium datasets...")

    for node_id in ["node_0", "node_1"]:
        node_cfg = config.consortium.nodes[node_id]
        n_samples = num_hub_samples
        node_seed = base_seed + int(node_id[-1]) * 101

        labels = create_dirichlet_skew(n_samples, pathologies, alpha, seed=node_seed)
        records = []

        for i in range(n_samples):
            sample_id = f"{node_id}_{i:05d}"
            img_path = synthetic_img_dir / f"{sample_id}.png"
            row_labels = labels[i]
            generate_chest_phantom(
                img_path,
                seed=node_seed + i,
                has_effusion=bool(row_labels[pathologies.index("Effusion")]),
                has_cardiomegaly=bool(row_labels[pathologies.index("Cardiomegaly")]),
                has_infiltration=bool(row_labels[pathologies.index("Infiltration")]),
                has_pneumonia=bool(row_labels[pathologies.index("Pneumonia")]),
            )
            rel_path = f"synthetic/{sample_id}.png"
            rec = {
                "sample_id": sample_id,
                "image_path": rel_path,
                "dataset": node_cfg["dataset"],
                "is_labeled": True,
            }
            for c_idx, p in enumerate(pathologies):
                rec[p] = int(row_labels[c_idx])
            records.append(rec)

        df = pd.DataFrame(records)
        split_file = splits_dir / f"{node_id.replace('node_', 'hub_')}.csv"
        df.to_csv(split_file, index=False)

        class_counts = {p: int(df[p].sum()) for p in pathologies}
        stats["nodes"][node_id] = {
            "name": node_cfg["name"],
            "role": node_cfg["role"],
            "sample_count": len(df),
            "class_distribution": class_counts,
        }
        logger.info(f"Saved {split_file.name} ({len(df)} samples)")

    for node_id in ["node_2", "node_3", "node_4", "node_5"]:
        node_cfg = config.consortium.nodes[node_id]
        n_samples = num_clinic_samples
        node_seed = base_seed + int(node_id[-1]) * 202

        labels = create_dirichlet_skew(n_samples, pathologies, alpha=0.5, seed=node_seed)
        records = []

        for i in range(n_samples):
            sample_id = f"{node_id}_{i:05d}"
            img_path = synthetic_img_dir / f"{sample_id}.png"
            row_labels = labels[i]
            generate_chest_phantom(
                img_path,
                seed=node_seed + i,
                has_effusion=bool(row_labels[pathologies.index("Effusion")]),
                has_cardiomegaly=bool(row_labels[pathologies.index("Cardiomegaly")]),
                has_infiltration=bool(row_labels[pathologies.index("Infiltration")]),
                has_pneumonia=bool(row_labels[pathologies.index("Pneumonia")]),
            )
            rel_path = f"synthetic/{sample_id}.png"
            rec = {
                "sample_id": sample_id,
                "image_path": rel_path,
                "dataset": node_cfg["dataset"],
                "is_labeled": False,
            }
            for c_idx, p in enumerate(pathologies):
                rec[p] = int(row_labels[c_idx])
            records.append(rec)

        df = pd.DataFrame(records)
        split_file = splits_dir / f"{node_id.replace('node_', 'clinic_')}.csv"
        df.to_csv(split_file, index=False)

        class_counts = {p: int(df[p].sum()) for p in pathologies}
        stats["nodes"][node_id] = {
            "name": node_cfg["name"],
            "role": node_cfg["role"],
            "sample_count": len(df),
            "class_distribution": class_counts,
        }
        logger.info(f"Saved {split_file.name} ({len(df)} samples)")

    val_labels = create_dirichlet_skew(num_val_samples, pathologies, alpha=1.0, seed=base_seed + 999)
    val_records = []
    for i in range(num_val_samples):
        sample_id = f"val_{i:05d}"
        img_path = synthetic_img_dir / f"{sample_id}.png"
        row_labels = val_labels[i]
        generate_chest_phantom(
            img_path,
            seed=base_seed + 999 + i,
            has_effusion=bool(row_labels[pathologies.index("Effusion")]),
            has_cardiomegaly=bool(row_labels[pathologies.index("Cardiomegaly")]),
            has_infiltration=bool(row_labels[pathologies.index("Infiltration")]),
            has_pneumonia=bool(row_labels[pathologies.index("Pneumonia")]),
        )
        rec = {
            "sample_id": sample_id,
            "image_path": f"synthetic/{sample_id}.png",
            "dataset": "nih_cxr14_val",
            "is_labeled": True,
        }
        for c_idx, p in enumerate(pathologies):
            rec[p] = int(row_labels[c_idx])
        val_records.append(rec)

    val_df = pd.DataFrame(val_records)
    val_file = splits_dir / "val_test.csv"
    val_df.to_csv(val_file, index=False)
    logger.info(f"Saved {val_file.name} ({len(val_df)} samples)")

    ext_labels = create_dirichlet_skew(num_ext_samples, pathologies, alpha=0.8, seed=base_seed + 888)
    ext_records = []
    for i in range(num_ext_samples):
        sample_id = f"ext_{i:05d}"
        img_path = synthetic_img_dir / f"{sample_id}.png"
        row_labels = ext_labels[i]
        generate_chest_phantom(
            img_path,
            seed=base_seed + 888 + i,
            has_effusion=bool(row_labels[pathologies.index("Effusion")]),
            has_cardiomegaly=bool(row_labels[pathologies.index("Cardiomegaly")]),
            has_infiltration=bool(row_labels[pathologies.index("Infiltration")]),
            has_pneumonia=bool(row_labels[pathologies.index("Pneumonia")]),
        )
        rec = {
            "sample_id": sample_id,
            "image_path": f"synthetic/{sample_id}.png",
            "dataset": "vindr_cxr",
            "is_labeled": True,
        }
        for c_idx, p in enumerate(pathologies):
            rec[p] = int(row_labels[c_idx])
        ext_records.append(rec)

    ext_df = pd.DataFrame(ext_records)
    ext_file = splits_dir / "external_test.csv"
    ext_df.to_csv(ext_file, index=False)
    logger.info(f"Saved {ext_file.name} ({len(ext_df)} samples)")

    bbox_records = []
    bbox_pathologies = ["Cardiomegaly", "Effusion", "Infiltration", "Pneumonia"]

    for i in range(num_bbox_samples):
        sample_id = f"bbox_{i:05d}"
        img_path = synthetic_img_dir / f"{sample_id}.png"
        chosen_finding = bbox_pathologies[i % len(bbox_pathologies)]

        if chosen_finding == "Cardiomegaly":
            x, y, w, h = 90, 120, 60, 50
        elif chosen_finding == "Effusion":
            x, y, w, h = 35, 150, 45, 30
        else:
            x, y, w, h = 50, 80, 40, 40

        generate_chest_phantom(
            img_path,
            seed=base_seed + 777 + i,
            has_effusion=(chosen_finding == "Effusion"),
            has_cardiomegaly=(chosen_finding == "Cardiomegaly"),
            has_infiltration=(chosen_finding == "Infiltration"),
            has_pneumonia=(chosen_finding == "Pneumonia"),
        )

        rec = {
            "sample_id": sample_id,
            "image_path": f"synthetic/{sample_id}.png",
            "finding": chosen_finding,
            "bbox_x": float(x),
            "bbox_y": float(y),
            "bbox_w": float(w),
            "bbox_h": float(h),
            "image_size": 224,
        }
        for p in pathologies:
            rec[p] = 1 if p == chosen_finding else 0
        bbox_records.append(rec)

    bbox_df = pd.DataFrame(bbox_records)
    bbox_file = splits_dir / "bbox_eval.csv"
    bbox_df.to_csv(bbox_file, index=False)
    logger.info(f"Saved {bbox_file.name} ({len(bbox_df)} samples)")

    stats_file = splits_dir / "consortium_stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved consortium stats to {stats_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FedMedVision Consortium Data Preparation")
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default="synthetic",
        help="Data preparation mode ('real' or 'synthetic')",
    )
    parser.add_argument(
        "--synthetic-dryrun",
        action="store_true",
        help="Fast synthetic dryrun mode producing instant phantoms and splits",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per hub in synthetic mode (default: 100)",
    )
    args = parser.parse_args()

    config = load_config()
    seed_everything(config.consortium.seeds[0] if config.consortium.seeds else 42)

    data_processed_dir = config.project_root / "data" / "processed"
    splits_dir = config.project_root / "data" / "splits"

    log_system_event(logger, "data_preparation_start", mode=args.mode, dryrun=args.synthetic_dryrun)

    if args.synthetic_dryrun or args.mode == "synthetic":
        build_synthetic_consortium(
            output_dir=data_processed_dir,
            splits_dir=splits_dir,
            config=config,
            num_hub_samples=args.num_samples,
            num_clinic_samples=int(args.num_samples * 0.8),
            num_val_samples=60,
            num_ext_samples=50,
            num_bbox_samples=30,
        )
    else:
        prepare_real_consortium(data_processed_dir, splits_dir, config)

    log_system_event(logger, "data_preparation_complete")
    print("✅ Consortium data preparation completed successfully.")


if __name__ == "__main__":
    main()
