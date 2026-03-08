# Pikachu Image Classifier using ResNet50 (PyTorch)

## Overview

This project implements an image classification model to detect **Pikachu** using a deep convolutional neural network. The model is built with **PyTorch** and uses a **ResNet50 architecture** with transfer learning from ImageNet weights.

The goal of the project is to train a model that can correctly classify whether an image contains Pikachu or not.

---
## Issues Encountered

* **DataLoader crashes** – Workers exited unexpectedly during training. Debugging required setting `num_workers=0` to surface the real error.
* **Corrupted images** – Some dataset images failed to load and caused runtime errors.
* **Tensor shape mismatches** – Image tensors in `[C,H,W]` format caused visualization errors, requiring conversion to `[H,W,C]` using `permute(1,2,0)`.
* **Class imbalance** – The dataset contained far fewer Pikachu images, which sometimes biased predictions toward the majority class.
* **Prediction verification** – Because Pikachu images were rare, random sampling and manual inspection were needed to confirm model behavior.
* **Training stability** – Batch losses occasionally fluctuated; this was mitigated using the AdamW optimizer and a cosine annealing learning rate scheduler.






---

## Model Architecture

The model uses **ResNet50**, a deep residual network that contains **50 layers** with skip connections to improve gradient flow during training.

Key components:

* Residual blocks
* Bottleneck layers
* Global average pooling
* Fully connected classification layer

The final classification layer is replaced to match the number of classes in the dataset.

---

## Training Configuration

**Loss Function**

```
CrossEntropyLoss
```

**Optimizer**

```
AdamW
lr = 3e-4
weight_decay = 1e-4
```

**Learning Rate Scheduler**

```
CosineAnnealingLR
T_max = 10
```

**Batch Size**

```
32
```

**Epochs**

```
50
```

---

## Dataset

The dataset should follow this structure:

```
dataset/
   train/
      pikachu/
      not_pikachu/
   val/
      pikachu/
      not_pikachu/
```

Training output example:

```
Epoch 1 Loss: 0.846
Epoch 2 Loss: 0.635
Epoch 3 Loss: 0.489
...
Epoch 10 Loss: 0.248
```

---

