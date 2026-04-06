# Whitebox Adversarial Attacks on YOLO-OBB: Evaluating FGSM Vulnerability

This repository documents experiments testing the foundational vulnerability of lightweight aerial detection models (`yolo26n-obb`) to first-order adversarial attacks, specifically utilizing the Fast Gradient Sign Method (FGSM) on the DOTA dataset.

---

## Part 1: Theory of First-Order Attacks (FGSM)

### Etymology
* **Fast:** The attack computes the necessary adversarial perturbation in a single forward and backward pass of the neural network. Because it does not rely on an iterative optimization loop, it is computationally inexpensive.
* **Gradient:** It utilizes the gradient (the first derivative) of the neural network's loss function with respect to the input image pixels.
* **Sign:** Instead of using the continuous magnitude (size) of the gradients, it extracts only the discrete *direction* (+1 or -1) to maximize the error uniformly across all dimensions.

### The Idea: The Linearity Hypothesis
Introduced by Ian Goodfellow et al. (2014) in the paper *"Explaining and Harnessing Adversarial Examples"*, FGSM challenged the assumption that neural networks fail due to high non-linearity. Instead, Goodfellow proved that they are vulnerable because they are **too linear** in high-dimensional spaces.

When a network processes a `1024x1024` color image, it is dealing with over 3 million individual dimensions (pixels). If an attacker shifts every single pixel by a tiny, imperceptible amount ($\epsilon$) in the exact direction of the gradient, the human eye sees no difference. However, because the network applies linear matrix multiplications across all 3 million pixels, those tiny changes *accumulate* into a massive, catastrophic shift in the final activation layers, forcing the model to confidently output the wrong prediction. FGSM acts as a blunt, single-step "sledgehammer" to exploit this high-dimensional linearity.

### The Formula
The mathematical representation of FGSM is a single-step optimization under an $L_\infty$ norm constraint:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

Where:
* $x$: The original, clean input image tensor.
* $x_{adv}$: The resulting adversarial image tensor.
* $\epsilon$: The perturbation budget. This enforces the $L_\infty$ norm constraint ($||x_{adv} - x||_\infty \le \epsilon$), ensuring no single pixel is altered by more than $\epsilon$.
* $J$: The specific loss function of the YOLO model (combining bounding box regression and classification loss).
* $\theta$: The frozen weights and parameters of the target model.
* $\nabla_x J$: The gradient of the loss with respect to the input image, obtained via backpropagation.
* $\text{sign}()$: A mathematical function mapping all positive gradient components to $1$, and negative components to $-1$, ensuring maximum perturbation at the edges of the $\epsilon$-ball.

---

## Part 2: Notebook Documentation (`fgsm_attack.ipynb`)

### Overview
This notebook establishes the baseline vulnerability of a Nano-scale Oriented Bounding Box (OBB) model (`yolo26n-obb`) to whitebox FGSM attacks. The experiment evaluates how targeted, single-step adversarial noise impacts the detection of complex, rotated objects (e.g., ships, planes, storage tanks) in high-resolution aerial imagery.

### Hyperparameters & System Configuration
The attack and evaluation pipeline were executed with the following precise configurations:
* **Target Model:** `yolo26n-obb.pt` (YOLOv8 Nano OBB architecture)
* **Dataset:** DOTA 1.5 (Dataset for Object Detection in Aerial Images)
* **Image Input Size:** `1024 x 1024` pixels
* **Perturbation Budgets ($\epsilon$):** Evaluated across a spectrum of strengths: `[0.01, 0.02, 0.04, 0.08]` (assuming pixel values normalized between `[0, 1]`)
* **Attack Iterations:** `1` (Standard Single-Step FGSM)
* **Target Classes:** 16 standard DOTA categories

### Experimental Pipeline
1. **Clean Baseline Establishment:** The pre-trained YOLO model is evaluated on standard, unperturbed `1024x1024` DOTA aerial images to record benchmark metrics (Precision, Recall, mAP50, and mAP95).
2. **FGSM Attack Generation:** A custom whitebox attacker is implemented. The script performs a forward pass to calculate the total OBB loss, runs a backward pass (`loss.backward()`) to track gradients back to the image pixels, extracts the sign of those gradients, multiplies by the strict $\epsilon$ budget, and adds it to the original image.
3. **Adversarial Evaluation:** The standard model is systematically evaluated against the newly generated adversarial datasets across the defined perturbation strengths.

### Key Findings & Results Interpretation
The generated plots document the rapid degradation of the model's predictive capabilities when exposed to single-step, first-order noise.

#### 1. Severe Accuracy Degradation
*(See `Plots/mAP50_mAP95.png` and `Plots/accuracy.png`)*
The results demonstrate a steep, negative correlation between the perturbation strength ($\epsilon$) and the model's Mean Average Precision. As $\epsilon$ increases, the accumulated linear noise successfully dismantles the network's spatial awareness. Even without multi-step optimization (like PGD), the single-step FGSM successfully exploits the Nano model's linear vulnerabilities, causing a massive drop in both mAP50 (general localization) and mAP95 (strict localization).

#### 2. The Mechanics of Failure: Cloaking vs. Hallucination
*(See `Plots/precision_recall.png`)*
The Precision and Recall curves illustrate exactly *how* the model fails under attack:
* **Recall Drop (Cloaking):** A sharp drop in Recall indicates that the attack successfully "cloaks" objects. The FGSM noise disrupts the high-frequency features (edges and corners) required for OBB regression, causing the model to completely miss highly visible targets.
* **Precision Drop (Hallucinations):** A simultaneous drop in Precision indicates that the mathematical noise tricks the model into drawing bounding boxes around empty space. The adversarial perturbation artificially inflates object-confidence scores in background regions, resulting in severe false positives.
