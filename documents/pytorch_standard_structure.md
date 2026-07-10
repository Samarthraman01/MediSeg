# PyTorch Standard Structure — Cheatsheet
> Every neural network project follows this exact pattern. Memorise this.

---

## The 8 Steps

### Step 1 — Imports
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
```
Bring in all libraries needed before anything else.

---

### Step 2 — Dataset Class
```python
class MyDataset(Dataset):
    def __init__(self, root, split='training'):
        # define paths
        # load list of filenames
        # define transforms

    def __len__(self):
        return len(self.images)   # total number of items

    def __getitem__(self, idx):
        # load one image from disk
        # load one mask/label from disk
        # apply transforms
        # return image, label
```
Defines HOW to load one image and its label.
- `__len__`  → how many items total
- `__getitem__` → load and return one item at index idx

---

### Step 3 — DataLoader
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=8,    # how many images per batch
    shuffle=True     # shuffle training data each epoch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False    # never shuffle test data
)
```
Wraps the dataset. Handles batching, shuffling, parallel loading.
DataLoader calls `__getitem__` automatically and stacks items into batches.

---

### Step 4 — Model Class
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define all layers here
        self.conv1 = nn.Conv2d(...)
        self.fc1   = nn.Linear(...)

    def forward(self, x):
        # define how data flows through layers
        x = self.conv1(x)
        x = self.fc1(x)
        return x
```
Defines the neural network architecture.
- `__init__`  → create all layers (runs once)
- `forward`   → define data flow (runs every batch)

---

### Step 5 — Setup
```python
device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model     = MyModel().to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
Create model instance, define loss function and optimizer, move to GPU.

**Loss functions:**
- `nn.MSELoss()`            → regression (predicting numbers)
- `nn.CrossEntropyLoss()`   → classification (predicting classes)
- `nn.BCELoss()`            → binary classification (yes/no)

**Optimizers:**
- `torch.optim.SGD(..., momentum=0.9)`  → simple, good for CNNs
- `torch.optim.Adam(...)`               → adaptive, good for complex models

---

### Step 6 — Training Loop
```python
for epoch in range(num_epochs):
    model.train()                          # set model to training mode

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass — get predictions
        predictions = model(images)

        # calculate loss — how wrong are we?
        loss = loss_fn(predictions, labels)

        # backward pass — calculate gradients
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # calculate new gradients
        optimizer.step()        # update weights

        if i % 50 == 0:
            print(f"Epoch {epoch+1} batch {i} — loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} complete — loss: {loss.item():.4f}")
```
The heart of training. Repeat forward → loss → backward → update for every batch.

**The four sacred lines (never change the order):**
```python
optimizer.zero_grad()   # 1. clear old gradients
loss.backward()         # 2. calculate new gradients
optimizer.step()        # 3. update weights
```

---

### Step 7 — Evaluation
```python
model.eval()                    # set model to evaluation mode

correct = 0
total   = 0

with torch.no_grad():           # no gradients needed for evaluation
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions       = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)

        correct += (predicted_classes == labels).sum().item()
        total   += labels.size(0)

accuracy = 100 * correct / total
print(f"Test accuracy: {accuracy:.2f}%")
```
Always call `model.eval()` before evaluating.
Always use `torch.no_grad()` — saves memory, runs faster.

**For segmentation — mIoU instead of accuracy:**
```python
intersection = torch.zeros(num_classes)
union        = torch.zeros(num_classes)

for cls in range(num_classes):
    pred_cls   = (predicted == cls)
    target_cls = (masks == cls)
    intersection[cls] += (pred_cls & target_cls).sum().item()
    union[cls]        += (pred_cls | target_cls).sum().item()

iou_per_class = intersection / (union + 1e-6)
miou = iou_per_class.mean().item()
```

---

### Step 8 — Save Model
```python
# save after training
torch.save(model.state_dict(), "models/my_model.pt")
print("Model saved!")

# load later for inference
model = MyModel().to(device)
model.load_state_dict(torch.load("models/my_model.pt", map_location=device))
model.eval()
```
`state_dict()` contains all the learned weights.
This file IS the trained model — share it, deploy it, load it anywhere.

---

## The Complete Flow

```
RAW DATA ON DISK
        ↓
Dataset class        Step 2 — loads one item at a time
        ↓
DataLoader           Step 3 — batches items together
        ↓
Model forward        Step 4 — processes each batch
        ↓
Loss function        Step 5 — measures how wrong
        ↓
Optimizer            Step 5 — updates weights
        ↓
Repeat               Step 6 — training loop
        ↓
Evaluate             Step 7 — measure final performance
        ↓
Save weights         Step 8 — store for deployment
```

---

## Quick Reference — What Changes Between Projects

| Component | What changes | What stays the same |
|-----------|-------------|---------------------|
| Dataset class | how images/masks are loaded | __len__ and __getitem__ structure |
| Model | architecture (layers) | nn.Module, __init__, forward |
| Loss function | depends on task | zero_grad → backward → step |
| Transforms | resize, normalize values | ToTensor, Normalize pattern |
| Metric | accuracy vs mIoU | evaluation loop structure |

---

## Results So Far

| Script | Task | Dataset | Result |
|--------|------|---------|--------|
| 01_neuron.py | single neuron | synthetic | loss → 0 |
| 02_layer.py | multi-layer | synthetic | forward pass verified |
| 03_activations.py | ReLU vs Sigmoid | - | visualised |
| 04_pytorch_intro.py | nn.Module | synthetic | model saved/loaded |
| 05_mnist.py | classification | MNIST 60k | 95.77% accuracy |
| 06_cnn_cifar.py | CNN classification | CIFAR-10 60k | 72.16% accuracy |
| 07_unet.py | segmentation arch | - | output (4,2,256,256) verified |
| 08_train_pets.py | binary segmentation | Oxford Pets 7k | loss 0.69→0.28 |
| 08_evaluate.py | mIoU evaluation | Oxford Pets 3.6k | mIoU 0.71 |

---

*MediSeg — github.com/Samarthraman01/MediSeg*
*Next: ADE20K pipeline → U-Net 150 classes → SegFormer fine-tune*
