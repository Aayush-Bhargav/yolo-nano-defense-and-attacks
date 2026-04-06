# Whitebox Adversarial Attacks on YOLO-OBB: Evaluating Quantized (Discrete) Attacks

This repository documents experiments testing the vulnerability of lightweight aerial detection models (`yolo26n-obb`) to real-world, format-resilient adversarial attacks, specifically utilizing **Quantized (Discretized) Iterative Optimization** on the DOTA dataset.

---

## Part 1: Theory of First-Order Attacks (Quantized/Discrete)

### Etymology
* **Quantized:** In digital signal processing, quantization is the process of mapping continuous, infinite values (like 32-bit floating-point numbers) into a smaller set of discrete, finite values (like 8-bit integers). 
* **Discrete:** The attack operates entirely within the constraints of discrete pixel steps (e.g., [0, 1, 2... 255]), rather than the continuous mathematical space of the neural network.

### The Idea: Solving the "Save-to-Disk" Problem
In standard academic settings, attacks like FGSM and PGD are calculated and applied entirely within the GPU using high-precision `float32` tensors. An optimal perturbation might dictate that a pixel's value should change by exactly $+0.00142$.

However, in real-world physical or digital attacks, adversarial images must eventually be saved as standard image files (e.g., PNG, JPG) to be fed into the target system. Standard images use `uint8` encoding, meaning pixels can only be whole numbers between 0 and 255. When a standard PGD `float32` image is saved to a hard drive, all of those delicate, high-precision decimal calculations are truncated or rounded off. This rounding acts as an unintentional defense mechanism, often completely destroying the efficacy of the adversarial noise.

**Quantized Attacks anticipate this.** Instead of calculating the perfect theoretical noise, a Quantized Attack forces the optimization loop to round its adversarial steps to the nearest valid 8-bit integer *during* the generation process. It ensures that the generated adversarial noise is perfectly survivable and remains fully lethal even after being saved to disk and re-loaded by a standard image parsing library (like OpenCV or PIL).

### The Formula
The mathematical representation of a Quantized Iterative Attack builds upon PGD, but introduces a non-differentiable Quantization step $Q(\cdot)$:

First, define the Quantization function for an 8-bit color space (where inputs are normalized between 0 and 1):
$$Q(z) = \frac{\text{round}(z \times 255)}{255}$$

The iterative optimization loop then becomes:
$$x_{adv}^{t+1} = Q \left( \Pi_{x+S} \left( x_{adv}^t + \alpha \cdot \text{sign}(\nabla_x J(\theta, x_{adv}^t, y)) \right) \right)$$

Where:
* $t$: The current iteration step.
* $x_{adv}^t$: The adversarial image at step $t$.
* $\alpha$: The step size. In quantized attacks, this is often strictly tied to integer pixel increments (e.g., $\alpha = 1/255$ or $2/255$).
* $J$: The YOLO-OBB loss function.
* $\nabla_x$: The gradient of the loss with respect to the input.
* $\Pi_{x+S}$: The projection operator maintaining the $\epsilon$-ball constraint.
* $Q(\cdot)$: The quantization operator forcing the intermediate tensor back into a valid, saveable discrete pixel space before the next iteration.

---

## Part 2: Notebook Documentation (`quantized_attacks.ipynb`)

### Overview
This notebook establishes the vulnerability of a Nano-scale Oriented Bounding Box (OBB) model (`yolo26n-obb`) to Quantized Iterative Attacks. The experiment evaluates whether the model remains vulnerable when the attacker is strictly constrained to physically realizable, 8-bit image formats, proving that adversarial vulnerabilities are not just theoretical `float32` artifacts.

### Hyperparameters & System Configuration
The attack and evaluation pipeline were executed with the following precise configurations:
* **Target Model:** `yolo26n-obb.pt` (YOLOv8 Nano OBB architecture)
* **Dataset:** DOTA 1.5 (Dataset for Object Detection in Aerial Images)
* **Image Input Size:** `1024 x 1024` pixels
* **Color Space Encoding:** `uint8` (0-255 scale, normalized to 0.0-1.0)
* **Perturbation Budgets ($\epsilon$):** Evaluated across continuous strengths `[0.01, 0.02, 0.04, 0.08]`, representing maximum pixel shifts of roughly `[3, 5, 10, 20]` out of 255.
* **Attack Iterations ($t$):** `10` discrete steps
* **Step Size ($\alpha$):** Dynamically scaled based on $\epsilon$ to ensure valid integer crossings.
* **Target Classes:** 16 standard DOTA categories

### Experimental Pipeline
1. **Clean Baseline Establishment:** The model is benchmarked on unperturbed, high-resolution aerial imagery to record ground-truth metrics.
2. **Quantized Attack Generation:** A custom iterative attacker is implemented. At each step, gradients are calculated, the step is taken, the noise is projected within the $\epsilon$ boundary, **and** the tensor is actively rounded to simulate 8-bit quantization. The final output is saved to disk as a standard image file.
3. **Adversarial Evaluation:** The images are loaded back from the hard drive (proving format resilience) and passed through the standard YOLO detection pipeline.

### Key Findings & Results Interpretation
The generated plots document the impact of format-resilient adversarial noise on the model's detection capabilities.

#### 1. Real-World Accuracy Degradation
*(See `Plots/mAP50_mAP95.png` and `Plots/accuracy.png`)*
The graphs reveal that restricting the attacker to discrete 8-bit pixel values does **not** protect the network. While the drop in Mean Average Precision might be marginally less steep than pure theoretical PGD (due to the loss of gradient precision during rounding), the model still suffers catastrophic degradation as $\epsilon$ increases. This proves that YOLO's vulnerability to adversarial examples can be easily exported into actual image files, making it a viable threat vector for deployed ML systems.

#### 2. The Mechanics of Failure in Discrete Space
*(See `Plots/precision_recall.png`)*
The Precision and Recall curves highlight how integer-constrained noise manipulates the detection head:
* **Recall Drop (Cloaking):** The quantized noise successfully destroys the network's object-proposal thresholds. By shifting pixel values by just a few integer steps (e.g., changing a pixel from RGB `120, 120, 120` to `115, 125, 115`), the edges of complex targets like ships and planes are completely masked from the network's convolutional filters.
* **Precision Drop (Hallucinations):** The attack successfully weaponizes discrete pixel blocks in the background to trigger false activations. The rounding function ensures these false features are harsh and distinct enough to survive image encoding, tricking the model into drawing high-confidence bounding boxes in empty space.