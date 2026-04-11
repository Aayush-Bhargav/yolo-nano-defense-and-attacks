# Defending YOLO-OBB: Adversarial Fine-Tuning against PGD

This repository documents experiments in hardening lightweight aerial detection models (`yolo26n-obb`) against whitebox adversarial attacks, specifically utilizing **Adversarial Training (PGD Defense)** on the DOTA dataset.

---

## Part 1: Theory of Adversarial Defense (PGD Training)

### Etymology
* **Adversarial Training:** The process of explicitly augmenting the model's training dataset with adversarial examples, forcing the model to learn robust feature representations rather than brittle, easily exploitable statistical artifacts.
* **PGD Defense:** Because Projected Gradient Descent (PGD) is considered the "universal" first-order adversary, training a model to resist PGD generally confers resistance to all other first-order attacks (like FGSM).

### The Idea: The Min-Max Saddle Point Problem
Introduced mathematically by Madry et al. (2017), adversarial training is best understood not as a simple data augmentation trick, but as a **robust optimization problem**.

Standard neural network training only attempts to minimize the loss over the natural training data. However, adversarial training frames the process as a two-player game:

1. **The Attacker (Inner Maximization):** For every image in the batch, the attacker actively explores the $\epsilon$-ball around the image to find the exact perturbation that *maximizes* the model's loss.
2. **The Defender (Outer Minimization):** The neural network then updates its weights via backpropagation to *minimize* the loss on those newly generated worst-case adversarial images.

By continually playing this game, the model is forced to abandon "weak" features and rely exclusively on "robust" features that the attacker cannot easily manipulate within the $\epsilon$ budget.

### The Formula
The mathematical representation of Adversarial Training is the Min-Max optimization formulation:

$$\min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{||\delta||_\infty \le \epsilon} L(\theta, x + \delta, y) \right]$$

Where:
* $\theta$: The trainable weights/parameters of the YOLO model.
* $\mathcal{D}$: The training dataset distribution (DOTA aerial images).
* $L$: The loss function of the model.
* $\max_{||\delta||_\infty \le \epsilon}$: The inner maximization problem (solved by running PGD to generate the worst-case perturbation $\delta$ within the allowed boundary).
* $\min_{\theta}$: The outer minimization problem (solved by the AdamW/SGD optimizer updating the model weights).

---

## Part 2: Notebook Documentation (`pgd-defense.ipynb`)

### Overview
This notebook implements a **Surgical Fine-Tuning** defense on the Nano-scale Oriented Bounding Box model (`yolo26n-obb`) to improve robustness against PGD attacks while preserving clean performance.

### Hyperparameters & System Configuration
* **Target Model:** `yolo26n-obb.pt` (YOLOv8 Nano OBB architecture)
* **Dataset:** Mixed Defense Dataset (50% Clean DOTA Images, 50% PGD-Attacked Images at $\epsilon=0.04$)
* **Image Input Size:** `1024 x 1024` pixels
* **Training Epochs:** `85` epochs
* **Batch Size:** `16` (Distributed Data Parallel via `device=[0, 1]`)
* **Learning Rate ($\text{lr0}$):** `0.0001`
* **Backbone Freezing:** `freeze=10`
* **Weight Decay:** `0.0005`

### Experimental Pipeline
1. **Defense Dataset Construction:** Balanced mix of clean aerial images and 10-step PGD adversarial examples.
2. **Surgical Fine-Tuning:** Frozen backbone + low learning rate to minimize catastrophic forgetting.
3. **Evaluation:** Head-to-head comparison of the defended model (`best.pt`) vs. the undefended baseline across PGD attacks ($\epsilon \in [0.01, 0.02, 0.04, 0.08]$).

### Key Findings & Results Interpretation
The evaluation plots show **clear robustness gains** from adversarial fine-tuning with **no evidence of catastrophic forgetting**.

#### 1. Clean Performance (No Forgetting – Slight Improvement)
*(See dashed lines in all plots: `f1Score.png`, `map50.png`, `map95.png`, `recall.png`, `precision.png`)*

The defended model actually outperforms the baseline on **clean** images:
- Higher F1 Score (~0.56 vs ~0.525)
- Higher mAP@50 (~0.455 vs ~0.435)
- Higher mAP@50-95 (~0.335 vs ~0.325)
- Higher Recall and Precision

This demonstrates that the surgical fine-tuning approach successfully hardened the model **without** sacrificing clean accuracy — a common pitfall avoided here thanks to the frozen backbone and micro learning rate.

#### 2. Robustness Under Attack (Strong Gains Across Most Metrics)
*(See solid lines in all plots)*

The defended model consistently outperforms the baseline attacked model across the entire $\epsilon$ range:

- **F1 Score** (`f1Score.png`): Defended attacked curve stays well above the baseline (e.g., 0.555 → 0.36 vs baseline 0.505 → 0.31).
- **mAP@50** (`map50.png`): Clear margin (0.455 → 0.275 vs 0.41 → 0.21).
- **mAP@50-95** (`map95.png`): Strong improvement (0.335 → 0.20 vs 0.305 → 0.15).
- **Recall** (`recall.png`): Better retention of true positives (0.425 → 0.28 vs 0.405 → 0.21).

#### 3. Precision Trade-off at High Perturbation Levels
*(See `precision.png`)*

- At low-to-moderate $\epsilon$ ($\leq 0.04$), the defended model shows **excellent precision gains** (starts at ~0.795), effectively suppressing hallucinations.
- At the highest perturbation ($\epsilon=0.08$), precision drops sharply for the defended model (~0.51), falling below the baseline attacked model. This indicates the model becomes overly conservative under extreme noise.

**Overall Verdict:** Adversarial fine-tuning delivered meaningful robustness improvements with **no clean-performance penalty** and strong gains in F1, mAP, and Recall. The only limitation appears at extreme $\epsilon$ levels in precision — a classic robustness–conservatism trade-off in low-capacity models.

---

**Plots included:** `f1Score.png`, `map50.png`, `map95.png`, `recall.png`, `precision.png`
