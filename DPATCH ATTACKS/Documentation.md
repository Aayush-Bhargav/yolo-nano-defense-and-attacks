# Adversarial Patch Attacks on YOLO-OBB: Evaluating DPatch Vulnerability

This repository documents experiments testing the vulnerability of a lightweight aerial detection model (`yolo26n-obb`) to physical-world adversarial patch attacks, specifically the **DPatch** method (Liu et al., 2019) on the DOTA dataset.

---

## Part 1: Theory of Adversarial Patch Attacks (DPatch)

### What is an Adversarial Patch?

Unlike pixel-level perturbation attacks (such as FGSM), which spread imperceptible noise across the entire input image, **adversarial patch attacks** concentrate all adversarial energy into a small, localized region of the image — a "patch". This patch is a learned pattern, optimized over many iterations, that is stamped directly onto an image. The key advantage over FGSM is that patches are **physically realizable**: they can theoretically be printed and placed in the real world to fool deployed detectors.

DPatch, introduced by Xin Liu et al. in *"DPatch: An Adversarial Patch Attack on Object Detectors"* (2019), was one of the first methods to demonstrate that a single, small, localized patch is sufficient to significantly degrade the performance of modern anchor-based and anchor-free object detectors — without requiring any knowledge of the scene or the target objects' locations.

### The Core Idea: Location-Agnostic Disruption

The central insight behind DPatch is that object detectors are not equally robust everywhere in their feature space. A neural network trained to detect objects must learn highly discriminative spatial features — edges, textures, corners — across the entire image. DPatch exploits this by learning a single patch pattern that, when placed anywhere in the image, disrupts these global feature representations enough to either:

1. **Suppress true detections** (cloaking): The patch poisons the feature maps such that the model fails to recognize objects it would otherwise detect confidently.
2. **Induce false positives** (hallucination): The adversarial pattern creates spurious activations in classification and regression heads, causing the model to predict bounding boxes in empty regions.

Unlike targeted attacks that aim to make the model predict a *specific* wrong class, DPatch is an **untargeted attack** — its only goal is to maximize the model's total loss, regardless of what wrong output it produces.

### The Formal Objective

For the untargeted variant of DPatch, the goal is to find a patch pattern $\hat{P}_u$ that maximizes the detection loss of the model when the patch is applied to input scenes. Formally:

$$\hat{P}_u = \arg\max_{P} \; \mathbb{E}_{x,s} \left[ \mathcal{L}\left( \mathcal{A}(x, s, P) \; \hat{y}, \hat{B} \right) \right]$$

Where:
- $x$: The original clean scene image.
- $s$: The patch placement parameters (position, scale, potentially rotation).
- $P$: The adversarial patch pattern being optimized.
- $\mathcal{A}(x, s, P)$: The "apply" function that stamps patch $P$ onto scene $x$ at location $s$, producing the adversarially patched image.
- $\hat{y}$: The true class label of objects in the scene.
- $\hat{B}$: The true bounding box labels of objects in the scene.
- $\mathcal{L}$: The object detector's loss function (combining classification and localization losses).

The expectation $\mathbb{E}_{x,s}$ is taken over a distribution of training images $x$ and patch placements $s$, ensuring the learned patch generalizes across diverse scenes and positions — making it robust and transferable.

### Why Patches Are Hard to Defend Against

A perturbation-based attack (like FGSM) is trivially defeated by JPEG compression or standard image preprocessing, which destroys the carefully crafted pixel-level noise. An adversarial patch, however, is **robust to such defenses** because it is a coherent, high-energy localized pattern — compressing or blurring an image does not eliminate a 150×150 pixel region of adversarial texture. This makes DPatch a more realistic threat model for deployed detection systems.

---

## Part 2: Notebook Documentation (`dpatch_attack.ipynb`)

### Overview

This notebook implements and evaluates the DPatch untargeted adversarial patch attack against a Nano-scale Oriented Bounding Box (OBB) detection model (`yolo26n-obb`) on high-resolution aerial imagery from the DOTA 1.5 dataset. The attack optimizes a fixed-size patch pattern over thousands of gradient-ascent iterations, with the trained patch subsequently evaluated against standard detection metrics.

### Hyperparameters & System Configuration

The attack and evaluation pipeline were executed with many combinations of hyperparameters, but the following precise configurations yielded the best results:

| Parameter | Value |
|---|---|
| **Target Model** | `yolo26n-obb.pt` (YOLOv8 Nano OBB architecture) |
| **Dataset** | DOTA 1.5 |
| **Image Input Size** | `640 × 640` pixels |
| **Patch Size** | `150 × 150` pixels |
| **Total Training Iterations** | `15,000` |
| **Evaluation Frequency** | Every `3,000` iterations |
| **Warm-up Iterations** | `500` |
| **Batch Size** | `4` |
| **Learning Rate** | `0.10` (Adam with AMSGrad, cosine annealing to `0.005`) |
| **TV Loss Weight** | `0.002` |
| **Max Patch Jitter** | `10 px` |
| **Training Images** | `1,600` |
| **Validation Images** | `400` |
| **Confidence Threshold (eval)** | `0.001` |
| **IoU Threshold (eval)** | `0.5` |
| **Random Seed** | `42` |
| **Platform** | Kaggle (T4 x 2) |

### Experimental Pipeline

#### 1. Clean Baseline Establishment
The pre-trained YOLO-OBB model is evaluated on `640×640` unperturbed DOTA aerial images, recording benchmark metrics: Precision, Recall, mAP50, and mAP50-95.

#### 2. Dataset Preparation
Source DOTA images and labels are split into training (1,600 images) and validation (400 images) subsets. DOTA annotation format (eight-coordinate polygon per object) is converted to the YOLO-OBB normalized format. Images are resized to `640×640` and YAML configuration files are generated for the Ultralytics evaluation pipeline.

#### 3. Loss Function Construction (Hook-Based)
A custom hook-based loss function is registered onto the final OBB detection head's `cv2` (box regression) and `cv3` (classification) convolutional branches. This gives direct gradient access to the raw pre-activation feature maps without interfering with the standard Ultralytics training API. The composite loss used for patch optimization consists of five terms:

- **Global class-confidence suppression**: Minimizes the mean squared sigmoid output of all classification feature maps across the entire image, pushing object confidence scores toward zero everywhere.
- **Patch-region entropy maximization**: Maximizes the binary entropy of classification logits within the patch footprint region of the feature map. High entropy corresponds to maximum uncertainty — the model becomes equally confused between all classes in the patch vicinity.
- **Box regression destruction**: Maximizes the mean absolute value and variance of raw box regression outputs, pushing the predicted geometry away from any coherent localization.
- **False-positive encouragement**: Adds a term that suppresses the mean predicted probability, indirectly encouraging noise-driven false activations in background regions.
- **Total variation regularization**: Penalizes high-frequency pixel discontinuities in the patch, producing a smoother, more physically plausible texture and improving print-world transferability.

#### 4. Patch Optimization Loop
At each iteration, a random batch of training images is loaded and the current patch is stamped onto each image at a randomly jittered top-left position. A forward pass through the model (in training mode, to keep BatchNorm statistics live) computes the composite adversarial loss. Backpropagation traces gradients back through the model to the patch tensor. The Adam optimizer (with AMSGrad) updates the patch, and pixel values are clamped to `[0, 1]` after each step. Checkpoints (patch tensor and training history) are saved every 1,000 iterations.

#### 5. Periodic Evaluation
Every 3,000 iterations, the current patch is written to disk and applied to the full validation set. Standard YOLO validation is run on both the clean and patched validation directories, and mAP50, mAP50-95, Precision, and Recall are recorded. Side-by-side visualizations of clean vs. patched detections are generated and saved.

#### 6. Final Evaluation & Summary
After 15,000 iterations, the final patch is applied to the validation set and full metrics are computed. A comparison table (Baseline vs. Attacked, with per-metric drop) is printed, and a training progress curve (mAP50, Precision, Recall vs. iteration) is saved.

---

## Part 3: Results & Analysis

### Key Findings

Despite ~30 hours of GPU compute on Kaggle (across multiple sessions) and extensive hyperparameter search, the DPatch attack produced only a modest degradation in model performance:

| Metric | Baseline | Attacked (DPatch) | Drop |
|---|---|---|---|
| **mAP50** | 0.41 | 0.36 | ▼ 0.05 (~12%) |
| **Precision** | 0.69 | 0.62 | ▼ 0.07 (~10%) |
| **Recall** | 0.37 | 0.33 | ▼ 0.04 (~11%) |

A ~10% relative decrease in mAP50 is **significantly below** the degradation levels reported by Liu et al. on PASCAL VOC / COCO-style detectors, indicating that the attack failed to generalize effectively in this setting.

---

## Part 4: Failure Analysis & Potential Reasons for Limited Effectiveness

### 1. Dataset Characteristics: High Object Density in DOTA

The most significant structural difference between this experiment and the original DPatch paper is the **object density** of the target dataset. DOTA aerial images routinely contain **dozens to hundreds of object instances per image** — ships, small vehicles, planes, and storage tanks that tile across the scene. A 150×150 pixel patch placed in one corner of a 640×640 image physically occludes only a tiny fraction of the image area. The model simply detects all the other objects it can still see, keeping overall mAP high. In contrast, the datasets used in the original paper (PASCAL VOC, MS COCO) feature fewer, larger, more centered objects — a single patch placement is far more disruptive when it covers a proportionally larger share of the relevant scene content.

### 2. Patch Size-to-Image Ratio

At `640×640` input resolution with a `150×150` patch, the adversarial region covers roughly **5.5% of the total image area**. For suppression-type attacks to work, the patch's adversarial signal must propagate through the feature pyramid and corrupt a substantial portion of the detection head's spatial outputs. Given DOTA's dense object distribution, the remaining ~94.5% of the image still contains abundant, clean feature signal for the model to detect objects from. Increasing the patch size (e.g., to 200×200 or 250×250) or using multiple patches per image would be natural next steps.

### 3. Hook-Based Loss vs. Native Training Loss

The loss function used in this implementation is a **proxy loss** constructed from raw feature map activations via forward hooks. It does not directly backpropagate through the model's native OBB loss computation (which includes Distribution Focal Loss for box regression and binary cross-entropy for classification, applied to matched anchors). The proxy objectives (suppressing sigmoid outputs, maximizing entropy) are reasonable approximations, but they may not generate gradients that most efficiently destroy the specific activations the detector relies on for OBB regression. Integrating the attack with Ultralytics' internal `compute_loss` function would create a tighter, more targeted gradient signal.

### 4. Model Architecture: Nano Scale and OBB Head

The `yolo26n-obb` Nano architecture is intentionally compact, with fewer parameters and narrower feature maps than standard or large-scale YOLO variants. Interestingly, this compactness may **reduce** susceptibility to patch attacks in some cases — the model has less redundant capacity for the patch to exploit, and the OBB regression head's geometric predictions (predicting angle in addition to box coordinates) may be less affected by global feature corruption than standard axis-aligned detection heads.

### 5. Training Instability and Resource Constraints

Several training sessions were interrupted due to Kaggle RAM and GPU memory overloads, particularly during the periodic evaluation phases when full validation sets were loaded alongside the training model. Interrupted sessions mean the patch optimization did not fully converge in some runs. EMA smoothing and checkpoint resumption mitigated this somewhat, but the effective number of clean gradient update steps was lower than the nominal 15,000 figure.

### 6. Fixed Patch Placement Strategy

During training, the patch was always placed in the top-left quadrant of the image (with small random jitter). This introduces a **placement bias**: the patch learns to disrupt features in the top-left spatial region of the feature pyramid but is not equally optimized for disrupting detections in the center or bottom-right of the image. Since DOTA images contain objects distributed across the entire scene, a patch that is not placement-agnostic will have reduced impact on the majority of objects outside its trained placement zone.

### 7. Hyperparameter Sensitivity

The experiments tested image sizes (`640×640` and `1024×1024`), patch sizes (80×80 to 160×160), learning rates (0.01 to 0.10), and batch sizes (1 to 16). No configuration produced a substantially stronger attack. This plateau across a wide hyperparameter range suggests the bottleneck is structural (dataset density, patch-to-image ratio) rather than tunable via standard hyperparameter search.

---

## Part 5: Directions for Future Work

* **Multiple simultaneous patches**: Tiling 3–5 patches across the image to maximize spatial coverage and increase the proportion of the scene that is adversarially corrupted.
* **Full OBB loss integration**: Bypassing the hook-based proxy and directly backpropagating through Ultralytics' native `compute_loss` for tighter gradient alignment.
* **Placement-agnostic training**: Randomizing patch placement across the full image (not just the top-left region) during optimization to produce a universally disruptive pattern.
* **Larger patch budgets**: Testing patch sizes above 200×200, or adaptive patches that scale with image resolution.
* **Iterative patch + global noise hybrid**: Combining DPatch with a mild global FGSM perturbation to attack objects outside the patch footprint simultaneously.
* **Targeted class suppression**: Rather than an untargeted loss, optimizing the patch to suppress detection of high-confidence classes (e.g., ships, small vehicles) specifically, which may produce stronger per-class drops even if overall mAP remains partially intact.
