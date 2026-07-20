import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Dataset ───────────────────────────────────────────────────────────
class ADE20KDataset(Dataset):
    def __init__(self, root, split='validation'):
        self.img_dir  = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'annotations', split)
        self.images   = sorted([
            f for f in os.listdir(self.img_dir) if f.endswith('.jpg')
        ])
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image    = Image.open(img_path).convert('RGB')
        image    = self.image_transform(image)

        mask_name = self.images[idx].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask      = Image.open(mask_path).resize((512, 512), Image.NEAREST)
        mask      = torch.from_numpy(np.array(mask)).long()
        mask      = mask.clamp(0, 149)

        return image, mask

# ── Setup ─────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Evaluating on: {device}", flush=True)

ROOT = os.path.expanduser('~/mediseg/data/ADEChallengeData2016')

val_dataset = ADE20KDataset(root=ROOT, split='validation')
val_loader  = DataLoader(val_dataset, batch_size=4, shuffle=False)
print(f"Validation images: {len(val_dataset)}", flush=True)

# ── Load model ────────────────────────────────────────────────────────
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(
    os.path.expanduser('~/mediseg/models/segformer_b2_epoch9.pt'),
    map_location=device
))
model = model.to(device)
model.eval()
print("Model loaded!", flush=True)

# ── mIoU calculation ──────────────────────────────────────────────────
intersection = torch.zeros(150)
union        = torch.zeros(150)

with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        masks  = masks.to(device)

        outputs   = model(pixel_values=images)
        predicted = torch.argmax(outputs.logits, dim=1)

        # upsample back to 512×512 — SegFormer outputs at lower resolution
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1).float(),
            size=(512, 512),
            mode='nearest'
        ).squeeze(1).long()

        for cls in range(150):
            pred_cls   = (predicted == cls)
            target_cls = (masks == cls)
            intersection[cls] += (pred_cls & target_cls).sum().item()
            union[cls]        += (pred_cls | target_cls).sum().item()

        if i % 50 == 0:
            print(f"Batch {i}/{len(val_loader)}", flush=True)

# ── Results ───────────────────────────────────────────────────────────
iou_per_class = intersection / (union + 1e-6)
miou = iou_per_class.mean().item()

print(f"\n{'='*50}")
print(f"Overall mIoU: {miou:.4f}")
print(f"{'='*50}")

# class names
class_names = [
    'wall','building','sky','floor','tree',
    'ceiling','road','bed','window','grass',
    'cabinet','sidewalk','person','earth','door',
    'table','mountain','plant','curtain','chair'
]
for i in range(20, 150):
    class_names.append(f'class_{i}')

iou_list   = [(class_names[i], iou_per_class[i].item()) for i in range(150)]
iou_sorted = sorted(iou_list, key=lambda x: x[1], reverse=True)

print("\nTop 10 best classes:")
for name, iou in iou_sorted[:10]:
    print(f"  {name:20s} IoU: {iou:.4f}")

print("\nBottom 10 worst classes:")
for name, iou in iou_sorted[-10:]:
    print(f"  {name:20s} IoU: {iou:.4f}")

# ── Comparison table ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print("FINAL COMPARISON")
print(f"{'='*50}")
print(f"U-Net from scratch:     mIoU = 0.0686")
print(f"SegFormer-B2 epoch 9:   mIoU = {miou:.4f}")
print(f"Improvement:            {miou/0.0686:.1f}× better")

# ── Bar chart ─────────────────────────────────────────────────────────
plt.figure(figsize=(20, 6))
ious   = [x[1] for x in iou_sorted]
colors = ['green' if iou > 0.4 else 'orange' if iou > 0.1 else 'red' for iou in ious]
plt.bar(range(150), ious, color=colors)
plt.axhline(y=miou, color='blue', linestyle='--', label=f'mIoU = {miou:.4f}')
plt.axhline(y=0.0686, color='red', linestyle='--', label='U-Net mIoU = 0.0686')
plt.xlabel('Class (sorted by IoU)')
plt.ylabel('IoU Score')
plt.title('SegFormer-B2 ADE20K — IoU per class')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.expanduser('~/mediseg/outputs/segformer_b2_miou.png'))
plt.show()
print("Chart saved to outputs/segformer_b2_miou.png")