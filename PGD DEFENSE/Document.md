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

By continually playing this game, the model is forced to abandon "weak" features (like high-frequency background textures that are easily corrupted by noise) and rely exclusively on "robust" features (like macroscopic shapes and deep semantic edges) that the attacker cannot easily manipulate within the $\epsilon$ budget.

### The Formula
The mathematical representation of Adversarial Training is the Min-Max optimization formulation:

$$\min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{||\delta||_\infty \le \epsilon} L(\theta, x + \delta, y) \right]$$

Where:
* $\theta$: The trainable weights/parameters of the YOLO model.
* $\mathcal{D}$: The training dataset distribution (DOTA aerial images).
* $L$: The loss function of the model.
* $\max_{||\delta||_\infty \le \epsilon}$: The inner maximization problem (solved by running PGD to generate the worst-case perturbation $\delta$ within the allowed boundary).
* $\min_{\theta}$: The outer minimization problem (solved by the AdamW/SGD optimizer updating the model weights to accurately predict bounding boxes despite the $\delta$ noise).

---

## Part 2: Notebook Documentation (`pgd-defense.ipynb`)

### Overview
This notebook attempts to armor a Nano-scale Oriented Bounding Box model (`yolo26n-obb`) against PGD attacks. Because low-capacity models are highly susceptible to "Catastrophic Forgetting" (overwriting their ability to see clean images in an attempt to block noise), this pipeline implements a **Surgical Fine-Tuning** approach, aggressively controlling the learning rate and freezing backbone layers.

### Hyperparameters & System Configuration
The defense training and evaluation pipeline were executed with the following precise configurations:
* **Target Model:** `yolo26n-obb.pt` (YOLOv8 Nano OBB architecture)
* **Dataset:** Mixed Defense Dataset (50% Clean DOTA Images, 50% PGD-Attacked Images at $\epsilon=0.04$)
* **Image Input Size:** `1024 x 1024` pixels
* **Training Epochs:** `30` epochs
* **Batch Size:** `16` (Distributed Data Parallel via `device=[0, 1]`)
* **Learning Rate ($\text{lr0}$):** `0.0001` (Micro learning rate to prevent violent weight destruction)
* **Warmup Epochs:** `3.0` (Slow ramp-up of gradients to avoid initial shock)
* **Backbone Freezing:** `freeze=10` (The first 10 layers, the model's "eyes", are locked to retain clean feature extraction)
* **Weight Decay:** `0.0005` (Regularization to prevent memorization of specific noise patterns)

### Experimental Pipeline
1. **Defense Dataset Construction:** A balanced dataset is dynamically generated containing clean aerial targets alongside mathematically poisoned targets generated via 10-step PGD.
2. **Surgical Fine-Tuning:** The model is trained on the mixed dataset. The frozen backbone ensures it remembers fundamental geometries, while the unfrozen "Neck" and "Head" logic layers attempt to learn noise-filtering algorithms.
3. **Ultimate Evaluation:** The newly defended model (`best.pt`) is pitted head-to-head against the standard, undefended model across a gauntlet of PGD attacks ($\epsilon \in [0.01, 0.02, 0.04, 0.08]$).

### Key Findings & Results Interpretation
The evaluation plots document the harsh realities of the Accuracy vs. Robustness Trade-off in low-capacity ML models.

#### 1. Catastrophic Forgetting & Capacity Limits
*(See `mAP50_mAP95.png` and `accuracy.png`)*
The defense successfully completed, but at a severe cost. The baseline clean accuracy (dashed lines) of the Defended model dropped significantly compared to the Regular model (e.g., mAP50 falling from ~0.65 to ~0.55). This mathematically proves that a "Nano" parameter space is insufficient to simultaneously store both complex OBB feature extractors *and* adversarial noise filters. The model had to overwrite good knowledge to accommodate the defense.

#### 2. Hallucination Resistance (The Silver Lining)
*(See `precision_recall.png`)*
Despite the drop in overall mAP, the defense *did* successfully alter the model's behavior under attack. 
* **Precision Gains:** At moderate perturbation levels ($\epsilon=0.02$), the Defended model maintained a much higher **Precision** than the Regular model. The Regular model succumbed to massive hallucinations (drawing false bounding boxes everywhere), whereas the Defended model correctly identified the mathematical noise and refused to trigger false positives.
* **Recall Losses:** The trade-off for this hallucination resistance was a heavily penalized Recall. The model became overly conservative, cloaking actual objects because it learned to distrust complex high-frequency textures.