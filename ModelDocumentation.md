# Target Architecture: YOLO-OBB Nano

This directory contains the pre-trained weights and configuration details for the target model evaluated throughout this repository. 

### The Model: `yolo26n-obb.pt`
The primary target for all adversarial experiments in this project is the **Nano variant** of the YOLO (You Only Look Once) architecture, specifically optimized for **Oriented Bounding Boxes (OBB)**. 

#### Why this specific architecture?
1. **Aerial Imagery Focus:** Standard object detectors draw axis-aligned bounding boxes, which fail catastrophically on densely packed, rotated objects often found in aerial/satellite imagery (e.g., ships docked at an angle, parked airplanes). OBB models regress an extra angle parameter ($\theta$) to draw tightly fitted, rotated boxes.
2. **The "Nano" Constraint:** The `n` (Nano) variant is intentionally chosen for its extremely low parameter count. Lightweight models are highly favored for deployment on edge devices (like drones or embedded flight systems). However, as our experiments prove, this lack of parameter capacity makes them disproportionately vulnerable to both adversarial attacks and "Catastrophic Forgetting" during defensive fine-tuning.

### Usage
To load and run baseline inference with this model, ensure you have the `ultralytics` package installed:

```python
from ultralytics import YOLO

# Load the target model
model = YOLO("model/yolo26n-obb.pt")

# Run inference on a sample DOTA image
results = model.predict(source="path/to/sample_image.png", imgsz=1024)

# Display results
results[0].show()