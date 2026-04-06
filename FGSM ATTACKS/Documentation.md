# Whitebox Adversarial Attacks on YOLO-OBB: Evaluating FGSM Vulnerability

This folder documents experiments testing the vulnerability of lightweight aerial detection models (`yolo26n-obb`) to first-order adversarial attacks, utilizing the DOTA (Dataset for Object Detection in Aerial Images) dataset.

---

## Part 1: Theory of First-Order Attacks (FGSM)

### Etymology
* **Fast:** The attack is computationally efficient. It computes the required perturbation in a single forward and backward pass, rather than running a lengthy, iterative optimization loop.
* **Gradient:** It utilizes the gradient (the first derivative) of the neural network's loss function with respect to the input image.
* **Sign:** Instead of using the actual magnitude (size) of the gradients, it extracts only the *direction* (+1 or -1) to maximize the error.

### The Idea
Introduced by Goodfellow et al. (2014), the Fast Gradient Sign Method (FGSM) operates on the principle that neural networks are highly linear in high-dimensional spaces. 

To force a model to make a mistake, the attacker calculates how the input image affects the model's loss. Then, every single pixel in the image is pushed one small step ($\epsilon$) in the exact direction that *increases* that loss the most. To the human eye, the image looks virtually unchanged, but the mathematical noise compounds within the network's matrix calculations, effectively blinding the model.

### The Formula
The mathematical representation of FGSM is:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

Where:
* $x$: The original, clean input image.
* $x_{adv}$: The resulting adversarial image.
* $\epsilon$: The perturbation budget (the maximum amount any pixel is allowed to change).
* $J$: The loss function of the model.
* $\theta$: The fixed weights/parameters of the trained model.
* $y$: The true ground-truth label of the image.
* $\nabla_x$: The gradient of the loss with respect to the input pixels.
* $\text{sign}()$: A mathematical function that converts all positive gradient numbers to 1, and negative numbers to -1.

---

## Part 2: Notebook Documentation (`fgsm_attack.ipynb`)

### Overview
This notebook establishes the baseline vulnerability of a Nano-scale Oriented Bounding Box (OBB) model (`yolo26n-obb`) to whitebox FGSM attacks. The experiment evaluates how targeted adversarial noise impacts the detection of complex, rotated objects (e.g., ships, planes, storage tanks) in aerial imagery.

### Experimental Pipeline
1. **Clean Baseline Establishment:** The pre-trained YOLO model is evaluated on standard, unperturbed DOTA aerial images to record its baseline metrics (Precision, Recall, mAP50, and mAP95).
2. **FGSM Attack Generation:** A custom whitebox attacker is implemented using the FGSM formula. It calculates the gradients of the model's loss function and applies targeted noise to the input images.
3. **Adversarial Evaluation:** The standard model is systematically evaluated against the newly generated adversarial datasets across varying perturbation strengths ($\epsilon$).

### Key Findings & Results Interpretation
The generated plots document the rapid degradation of the model's predictive capabilities when exposed to first-order noise.

#### 1. Severe Accuracy Degradation
*(See `Plots/mAP50_mAP95.png` and `Plots/accuracy.png`)*
The results demonstrate a steep, negative correlation between the perturbation strength ($\epsilon$) and the model's Mean Average Precision. As $\epsilon$ increases, the FGSM noise successfully dismantles the network's ability to draw accurate bounding boxes, proving that lightweight aerial detection models are highly susceptible to gradient-based pixel manipulation.

#### 2. The Mechanics of Failure: Cloaking vs. Hallucination
*(See `Plots/precision_recall.png`)*
The Precision and Recall curves illustrate exactly *how* the model fails under attack:
* **Recall Drop (Cloaking):** A sharp drop in Recall indicates that the attack successfully "cloaks" objects. The model completely misses highly visible targets like ships or planes.
* **Precision Drop (Hallucinations):** A simultaneous drop in Precision indicates that the mathematical noise tricks the model into drawing bounding boxes around empty space or background noise, falsely identifying objects that do not exist.