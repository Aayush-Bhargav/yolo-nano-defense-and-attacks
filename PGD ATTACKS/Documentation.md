# Whitebox Adversarial Attacks on YOLO-OBB: Evaluating PGD Vulnerability

This repository documents experiments testing the vulnerability of lightweight aerial detection models (`yolo26n-obb`) to advanced first-order adversarial attacks, specifically utilizing Projected Gradient Descent (PGD) on the DOTA dataset.

---

## Part 1: Theory of First-Order Attacks (PGD)

### Etymology
* **Projected:** The attack operates under a strict mathematical constraint known as the $\epsilon$-ball (usually defined by the $L_\infty$ norm). If an iterative optimization step pushes a pixel's value outside this allowed boundary, the value is mathematically "projected" (clipped) back to the boundary's edge.
* **Gradient Descent:** It utilizes gradient optimization techniques. (Note: In the context of adversarial attacks, it is technically gradient *ascent*, as the objective is to maximize the model's loss rather than minimize it).

### The Idea: Constrained Optimization in High-Dimensional Space
Introduced by Madry et al. (2017) in their seminal paper *"Towards Deep Learning Models Resistant to Adversarial Attacks"*, Projected Gradient Descent (PGD) is widely considered the ultimate "universal" first-order adversary. 

To understand PGD, we must view adversarial attacks as a **constrained optimization problem**. We want to find a perturbation $\delta$ that maximizes the network's loss $L(x + \delta, y)$, subject to the constraint that the perturbation remains invisible to the human eye ($||\delta||_\infty \le \epsilon$).

While single-step attacks like FGSM assume the neural network's decision boundary is perfectly linear, deep neural networks are highly non-linear. FGSM often "overshoots" or misses the optimal point of failure. PGD solves this by acting as a "scalpel." It takes many tiny steps ($\alpha$), recalculating the local gradient at each step. By iteratively navigating the complex, non-linear loss landscape, PGD consistently converges on the absolute worst-case perturbation, making it a much more lethal and reliable attack than FGSM.

### The Formula
The mathematical representation of PGD is applied in a loop over $t$ iterations:

$$x_{adv}^{t+1} = \Pi_{x+S} \left( x_{adv}^t + \alpha \cdot \text{sign}(\nabla_x J(\theta, x_{adv}^t, y)) \right)$$

Where:
* $t$: The current iteration step.
* $x_{adv}^t$: The adversarial image at step $t$ (starts as the clean image $x$ at $t=0$, often initialized with random uniform noise to avoid local minima).
* $\alpha$: The step size (hyperparameter dictating how far to move per iteration).
* $J$: The loss function of the target model.
* $\theta$: The fixed weights/parameters of the trained model.
* $\nabla_x$: The gradient of the loss with respect to the input pixels.
* $\text{sign}()$: Extracts the direction of the gradient (+1 or -1).
* $\Pi_{x+S}$: The projection operator ensuring the cumulative noise stays within the allowed limit $S$ (the $\epsilon$-ball constraint, ensuring $x_{adv}$ remains within $[x-\epsilon, x+\epsilon]$ and valid image ranges $[0, 1]$ or $[0, 255]$).

---

## Part 2: Notebook Documentation (`pgd_attack.ipynb`)

### Overview
This notebook establishes the baseline vulnerability of a Nano-scale Oriented Bounding Box (OBB) model (`yolo26n-obb`) to iterative whitebox PGD attacks. The experiment evaluates how this highly optimized, multi-step adversarial noise impacts the detection of complex, rotated objects in aerial imagery.

### Hyperparameters & System Configuration
The attack and evaluation pipeline were executed with the following precise configurations:
* **Target Model:** `yolo26n-obb.pt` (YOLOv8 Nano OBB)
* **Dataset:** DOTA 1.5 (Dataset for Object Detection in Aerial Images)
* **Image Input Size:** `1024 x 1024` pixels
* **Perturbation Budgets ($\epsilon$):** Evaluated at multiple strengths: `[0.01, 0.02, 0.04, 0.08]`
* **PGD Iterations ($t$):** `10` steps per image
* **PGD Step Size ($\alpha$):** Dynamically calculated as $\epsilon \times 0.25$
* **Target Classes:** 16 standard DOTA categories (planes, ships, storage tanks, large/small vehicles, etc.)

### Experimental Pipeline
1. **Clean Baseline Establishment:** The pre-trained YOLO model is evaluated on standard, unperturbed `1024x1024` DOTA aerial images to record benchmark metrics (Precision, Recall, mAP50, and mAP95).
2. **Iterative Attack Generation:** A custom whitebox attacker is implemented using the PGD formula. For 10 iterations, the script computes the forward pass, calculates the loss across all bounding box predictions, backpropagates to find the input gradients, steps in the direction of the gradient by $\alpha$, and projects the pixels back into the valid $\epsilon$-ball.
3. **Adversarial Evaluation:** The standard model is systematically evaluated against the newly generated adversarial datasets across the four defined perturbation strengths.

### Key Findings & Results Interpretation
The generated plots document the severe degradation of the model's predictive capabilities when exposed to iteratively optimized noise.

#### 1. Severe Accuracy Degradation
*(See `mAP50_mAP95.png` and `accuracy.png`)*
The results demonstrate a catastrophic drop in the model's Mean Average Precision. Even at low perturbation budgets ($\epsilon = 0.02$), the PGD attack successfully exploits the deepest vulnerabilities in the network's feature extraction backbone. Because PGD optimizes the noise over 10 discrete steps, it bypasses the linear assumptions of the network, causing the mAP50 and mAP95 metrics to plummet far more aggressively than single-step attacks, effectively neutralizing the model's ability to detect oriented bounding boxes.

#### 2. The Mechanics of Failure: Cloaking vs. Hallucination
*(See `precision_recall.png`)*
The Precision and Recall curves detail the specific failure modes induced by the optimized PGD attack:
* **Recall Drop (Cloaking):** The sharp, near-vertical decrease in Recall indicates that the PGD attack is highly successful at cloaking targets. The optimized noise breaks the local texture and edge features relied upon by the YOLO detection head, causing the model to completely ignore highly visible targets in the aerial imagery.
* **Precision Drop (Hallucinations):** The concurrent drop in Precision indicates that the iterative noise acts as a potent false-positive generator. By maximizing the loss function, the network is tricked into identifying the mathematical perturbation patterns as actual object features, drawing highly confident but completely false bounding boxes around empty background space.