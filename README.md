# Robust Aerial Detection: Adversarial Vulnerabilities in YOLO-OBB

Detecting targets in high-resolution aerial imagery is a critical task for autonomous flight and satellite surveillance. However, the lightweight edge-models often deployed for these tasks harbor severe mathematical vulnerabilities.

This repository explores the adversarial robustness of Nano-scale Oriented Bounding Box (OBB) object detectors. It documents the catastrophic failure modes of `YOLO-OBB` when exposed to various first-order digital adversarial attacks.

## 📂 Repository Structure

```text
yolo-adversarial-robustness/
│
├── FGSM ATTACKS/                  # Single-step global adversarial attacks
│   ├── Plots/                     # Evaluation visualizations (Accuracy, mAP, PR curves)
│   ├── Documentation.md           # Deep-dive theory on Goodfellow's Linearity Hypothesis
│   └── fgsm_attack.ipynb          # Baseline FGSM pipeline
│
├── PGD ATTACKS/                   # Iterative constrained optimization attacks
│   ├── Plots/                     # Evaluation visualizations (Accuracy, mAP, PR curves)
│   ├── Documentation.md           # Deep-dive theory on Min-Max saddle point optimization
│   └── pgd_attack.ipynb           # Multi-step PGD pipeline
│
├── PGD DEFENSE/                   # Defense strategies against PGD attacks
│   ├── Document.md                # Defense documentation and analysis                     
|   ├── Plots/                     # Evaluation visualizations (Accuracy, mAP, PR curves)
│   └── pgd-defense.ipynb          # PGD defense implementation and evaluation
|   └── plot_generation_code.ipynb # Code for the generation of the plots
│   └── best.pt                    # Model trained for 85 epochs
|
├── QUANTIZED ATTACKS/             # Attacks on quantized (edge-optimized) models
│   ├── Plots/                     # Evaluation visualizations (Accuracy, mAP, PR curves)
│   ├── Documentation.md           # Theory and results for quantized-model attacks
│   └── quantized_attacks.ipynb    # Quantized attack generation and evaluation
│
└── ModelDocumentation.md          # Overall model details and baseline performance
