# FedMedVision — Persistent Workspace Rules (AGENT.md)

> These rules are **binding for every phase and every future task** in this project. No agent, script, or human contributor may override them without an explicit versioned amendment.

---

## 1. Hardware & VRAM Budget

| Constraint | Value |
|---|---|
| Target GPU | HP Victus laptop GPU (≤ 6 GB VRAM) |
| Fallback | CPU (automatic) |
| Mixed Precision | `torch.cuda.amp.autocast` (FP16) **always enabled** |
| Max Batch Size | `batch_size <= 32` (default `16`) |
| Data Workers | `num_workers <= 2` |
| FL Runtime | Flower Simulation mode (`flwr[simulation]`) for all training & ablation |

---

## 2. Core Stack Versions

| Package | Version Constraint |
|---|---|
| Python | `>=3.11` |
| torch / torchvision | latest stable (CUDA 12.x wheel) |
| timm | `>=1.0.9` |
| flwr[simulation] | `>=1.43.0` |
| opacus | `>=1.6.0` |
| grad-cam | `>=1.5.4` |
| albumentations | `>=1.4.18` |
| nvidia-ml-py (pynvml) | latest |
| scikit-learn | `>=1.5.0` |
| pandas | `>=2.2.0` |
| numpy | `>=1.26.0` |
| pyyaml | `>=6.0.1` |
| fastapi | `>=0.115.0` |
| uvicorn[standard] | `>=0.30.0` |
| reportlab | `>=4.2.0` |

---

## 3. System Architecture Contract

### 6-Node Medical Consortium

| Node | Role | Dataset | Scans | Distribution |
|---|---|---|---|---|
| 0 | Labeled Urban Hub | NIH ChestX-ray14 | 8,000 | Dirichlet α=0.3 |
| 1 | Labeled Urban Hub | NIH ChestX-ray14 | 8,000 | Dirichlet α=0.3 |
| 2 | Unlabeled Regional Clinic | CheXpert | 10,000 | MC-Dropout pseudo-labels |
| 3 | Unlabeled Regional Clinic | CheXpert | 10,000 | MC-Dropout pseudo-labels |
| 4 | Unlabeled Rural Clinic | CheXpert | 10,000 | MC-Dropout pseudo-labels |
| 5 | Unlabeled Rural Clinic + **Byzantine Poisoning Node** | CheXpert | 10,000 | MC-Dropout pseudo-labels; toggleable poisoning |

### Model Architecture
- Backbone: `convnext_tiny` (ImageNet pretrained)
- Private local normalization: `FedBN` state-dict filtering (norm layers excluded from aggregation)
- MC-Dropout: `p=0.2`, applied during inference for uncertainty estimation

### Optimization Stack
- Loss: `BCEWithLogitsLoss` (multi-label, 5 pathologies)
- Federated: `FedProx` (mu=0.01) + `FedProto` class-wise latent prototype alignment
- Green AI: Stage-wise gradient-norm adaptive freezing

### Security & Privacy
- **Differential Privacy:** Opacus DP-SGD, epsilon sweep {1.0, 3.0, 8.0, ∞}, delta=1e-5
- **Secure Aggregation:** Flower `SecAgg+`
- **Byzantine Defense:** Coordinate-wise Trimmed Mean (trim_ratio=0.15)

### XAI & Calibration
- Explainability: `Grad-CAM++` evaluated against NIH BBox_List_2017.csv (IoU metric)
- Calibration: Post-hoc Temperature Scaling (Expected Calibration Error — ECE)

---

## 4. Verification Law

**Every phase MUST end by creating and running `tests/test_phaseX.py` via pytest, achieving 100% pass rate before the phase is considered complete.** Fix all errors before concluding.

---

## 5. Pathologies

`["Cardiomegaly", "Effusion", "Infiltration", "Pneumonia", "No Finding"]`

---

## 6. File Layout

```
fedmedvision/
├── AGENT.md                  # This file — workspace rules
├── configs/                  # YAML configuration files
├── data/                     # Raw, processed, splits
├── models/                   # Model definitions
├── security/                 # DP, SecAgg, Byzantine defense
├── fl_engine/                # Flower strategies & simulation
├── app/                      # FastAPI backend + frontend
├── experiments/              # Experiment scripts & results
├── results/                  # Checkpoints, logs, metrics
├── paper/                    # Figures and paper assets
├── utils/                    # Config loader, reproducibility, logger
└── tests/                    # Phase smoke tests
```
