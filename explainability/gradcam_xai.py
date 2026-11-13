"""
FedMedVision — Explainability (Grad-CAM++ & Real Radiologist BBox IoU Validation)
Targeting ConvNeXt backbone stage 3 blocks for diagnostic localization:
- Grad-CAM++ activation maps for single-image diagnosis with MC-Dropout uncertainty
- Evaluation against NIH BBox_List_2017.csv (Mean IoU and Pointing Game Accuracy)
- High-resolution visual comparisons saved to results/xai_overlays/
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from typing import Any

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data.dataset import get_medical_transforms
from models.convnext_fedbn import FedMedConvNeXt
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("fedmedvision.xai")


class BorderFreeGradCAMPlusPlus(GradCAMPlusPlus):
    """
    Subclass of GradCAMPlusPlus that suppresses ConvNeXt zero-padding edge artifacts
    before upsampling the raw 7x7 CAM activation map to 224x224.
    """

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: list[Any], eigen_smooth: bool
    ) -> list[np.ndarray]:
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)

            # Suppress ConvNeXt zero-padding edge artifacts on raw 7x7 feature map
            for b in range(cam.shape[0]):
                cam[b, 0, :] *= 0.05
                cam[b, -1, :] *= 0.05
                cam[b, :, 0] *= 0.05
                cam[b, :, -1] *= 0.05
                # Re-normalize to [0, 1]
                c_min, c_max = np.min(cam[b]), np.max(cam[b])
                cam[b] = (cam[b] - c_min) / (1e-7 + (c_max - c_min))

            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer


def create_clean_cam_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    min_threshold: float = 0.25,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    """
    Creates a clean border-free Grad-CAM++ overlay where healthy background tissue (< min_threshold)
    remains clean grayscale while only true lung lesions glow vividly.
    """
    clean_mask = mask.copy()
    clean_mask[clean_mask < min_threshold] = 0.0
    # Re-normalize above threshold to maintain vivid color gradient
    active_pts = clean_mask >= min_threshold
    if np.any(active_pts):
        clean_mask[active_pts] = (clean_mask[active_pts] - min_threshold) / (1.0 - min_threshold + 1e-7)

    heatmap = cv2.applyColorMap(np.uint8(255 * clean_mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0

    # Alpha blend only where clean_mask > 0; leave background completely untouched
    alpha = (clean_mask[:, :, None] > 0.0).astype(np.float32) * (1.0 - image_weight)
    overlay = (1.0 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return np.uint8(255 * overlay)


class MedicalGradCAMExplainer:
    """
    Grad-CAM++ Explainability Engine for FedMedConvNeXt.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.model.eval()

        if target_layer is None:
            # Default target layer: last block of stage 3
            self.target_layer = [self.model.backbone.stages[-1].blocks[-1]]
        else:
            self.target_layer = [target_layer]

        self.cam = BorderFreeGradCAMPlusPlus(
            model=self.model,
            target_layers=self.target_layer,
        )
        self.eval_transform = get_medical_transforms(image_size=224, mode="eval")
        self.config = load_config()
        self.pathologies = self.config.consortium.pathologies

    def explain_single_image(
        self,
        image_path_or_pil: str | Path | Image.Image,
        target_class_idx: int | None = None,
    ) -> dict[str, Any]:
        """
        Runs single chest X-ray diagnosis with 5-pass MC-Dropout uncertainty
        and generates Grad-CAM++ heatmap and RGB overlay.
        """
        if isinstance(image_path_or_pil, (str, Path)):
            p = Path(image_path_or_pil)
            if not p.is_absolute():
                p = self.config.project_root / "data" / "processed" / p
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {p}")
            with Image.open(p) as img:
                pil_img = img.convert("RGB").resize((224, 224))
        else:
            pil_img = image_path_or_pil.convert("RGB").resize((224, 224))

        rgb_np = np.array(pil_img)
        norm_float_img = rgb_np.astype(np.float32) / 255.0

        augmented = self.eval_transform(image=rgb_np)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        # 1. Prediction & MC-Dropout Uncertainty
        with torch.no_grad():
            mean_probs, entropy = self.model.predict_with_uncertainty(input_tensor, num_passes=5)
            probs = mean_probs[0].cpu().numpy()
            uncertainties = entropy[0].cpu().numpy()

        probs_dict = {p_name: round(float(probs[i]), 4) for i, p_name in enumerate(self.pathologies)}
        uncertainty_dict = {p_name: round(float(uncertainties[i]), 4) for i, p_name in enumerate(self.pathologies)}

        # Multi-label clinical decision logic:
        # Detect any active pathology above clinical operating threshold (0.50)
        # Exclude index 4 ("No Finding") when checking for active lesions
        pathology_indices = [i for i, p in enumerate(self.pathologies) if p != "No Finding"]
        active_pathologies = [(i, probs[i]) for i in pathology_indices if probs[i] >= 0.50]

        if target_class_idx is None:
            if active_pathologies:
                # Select the highest-confidence detected disease pathology
                target_class_idx = max(active_pathologies, key=lambda x: x[1])[0]
            else:
                # If no specific disease crosses threshold, check if No Finding is dominant
                no_finding_idx = self.pathologies.index("No Finding") if "No Finding" in self.pathologies else -1
                if no_finding_idx != -1 and probs[no_finding_idx] >= 0.50:
                    target_class_idx = no_finding_idx
                else:
                    # Otherwise target highest disease finding
                    target_class_idx = int(np.argmax(probs[:4])) if len(probs) >= 4 else int(np.argmax(probs))

        pred_class_name = self.pathologies[target_class_idx]

        # Overall epistemic uncertainty is the mean entropy across all pathologies
        mean_predictive_entropy = float(np.mean(uncertainties))

        # 2. Grad-CAM++ Localization
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]  # (224, 224)

        heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)
        cam_overlay = create_clean_cam_overlay(norm_float_img, grayscale_cam, min_threshold=0.25)

        return {
            "probs": probs_dict,
            "uncertainty": uncertainty_dict,
            "predictive_entropy": round(mean_predictive_entropy, 4),
            "predicted_class": pred_class_name,
            "target_class_idx": target_class_idx,
            "heatmap": heatmap_uint8,
            "cam_overlay": cam_overlay,
            "pil_image": pil_img,
        }

    def evaluate_bbox_localization(
        self,
        bbox_csv: str | Path = "data/splits/bbox_eval.csv",
        threshold: float = 0.5,
        output_dir: str | Path = "results/xai_overlays",
    ) -> dict[str, Any]:
        """
        Evaluates localization against radiologist annotations in BBox_List_2017.csv:
        - Mean IoU of CAM thresholded mask against radiologist ground-truth box
        - Pointing Game Accuracy (whether peak CAM activation falls inside radiologist box)
        """
        bbox_path = Path(bbox_csv)
        if not bbox_path.is_absolute():
            bbox_path = self.config.project_root / bbox_path

        df_bbox = pd.read_csv(bbox_path)
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            out_path = self.config.project_root / out_path
        out_path.mkdir(parents=True, exist_ok=True)

        ious = []
        pointing_hits = 0
        total_eval = 0

        saved_overlay_paths = []

        for idx, row in df_bbox.iterrows():
            img_rel = str(row["image_path"])
            finding = str(row["finding"])

            if finding not in self.pathologies:
                continue

            target_idx = self.pathologies.index(finding)

            try:
                explanation = self.explain_single_image(img_rel, target_class_idx=target_idx)
            except FileNotFoundError:
                continue

            cam_mask = explanation["heatmap"] / 255.0  # (224, 224)

            # Ground truth box [x, y, w, h] at 224x224
            bx = float(row["bbox_x"])
            by = float(row["bbox_y"])
            bw = float(row["bbox_w"])
            bh = float(row["bbox_h"])

            gt_mask = np.zeros((224, 224), dtype=bool)
            x_min = max(0, int(bx))
            y_min = max(0, int(by))
            x_max = min(224, int(bx + bw))
            y_max = min(224, int(by + bh))
            gt_mask[y_min:y_max, x_min:x_max] = True

            # Thresholded CAM mask
            bin_cam = cam_mask >= threshold

            intersection = np.logical_and(bin_cam, gt_mask).sum()
            union = np.logical_or(bin_cam, gt_mask).sum()
            iou = float(intersection) / max(1, union)
            ious.append(iou)

            # Pointing game hit: peak CAM location
            peak_y, peak_x = np.unravel_index(np.argmax(cam_mask), cam_mask.shape)
            is_hit = bool(gt_mask[peak_y, peak_x])
            if is_hit:
                pointing_hits += 1
            total_eval += 1

            # Save first 6 side-by-side comparison overlays
            if len(saved_overlay_paths) < 6:
                overlay_img = Image.fromarray(explanation["cam_overlay"])
                draw = ImageDraw.Draw(overlay_img)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="lime", width=3)
                # Draw small circle for peak
                draw.ellipse([peak_x - 3, peak_y - 3, peak_x + 3, peak_y + 3], fill="red", outline="white")

                fname = f"bbox_overlay_{idx:02d}_{finding.lower()}.png"
                save_file = out_path / fname
                overlay_img.save(save_file)
                saved_overlay_paths.append(str(save_file))

        mean_iou = float(np.mean(ious)) if ious else 0.0
        pointing_acc = (float(pointing_hits) / max(1, total_eval)) if total_eval > 0 else 0.0

        return {
            "total_evaluated": total_eval,
            "mean_iou": round(mean_iou, 4),
            "pointing_game_accuracy": round(pointing_acc, 4),
            "saved_overlays": saved_overlay_paths,
        }

