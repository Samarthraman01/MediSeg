import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import onnxruntime as ort

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
        img_path  = os.path.join(self.img_dir, self.images[idx])
        image     = Image.open(img_path).convert('RGB')
        image     = self.image_transform(image)

        mask_name = self.images[idx].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask      = Image.open(mask_path).resize((512, 512), Image.NEAREST)
        mask      = torch.from_numpy(np.array(mask)).long()
        mask      = mask.clamp(0, 149)

        return image, mask

# ── Setup ─────────────────────────────────────────────────────────────
ROOT = os.path.expanduser('~/mediseg/data/ADEChallengeData2016')

val_dataset = ADE20KDataset(root=ROOT, split='validation')
val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"Validation images: {len(val_dataset)}")

# load INT8 model
session = ort.InferenceSession(
    os.path.expanduser('~/mediseg/models/segformer_b2_int8.onnx')
)
print("INT8 model loaded!")

# ── mIoU calculation ──────────────────────────────────────────────────
intersection = torch.zeros(150)
union        = torch.zeros(150)

for i, (images, masks) in enumerate(val_loader):
    # run ONNX inference
    onnx_input = images.numpy()
    logits     = session.run(None, {"pixel_values": onnx_input})[0]
    logits     = torch.from_numpy(logits)

    # upsample to 512×512
    predicted = torch.argmax(logits, dim=1)
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

    if i % 100 == 0:
        print(f"Batch {i}/{len(val_loader)}")

iou_per_class = intersection / (union + 1e-6)
miou = iou_per_class.mean().item()

print(f"\n── INT8 Results ──")
print(f"INT8 mIoU:  {miou:.4f}")
print(f"FP32 mIoU:  0.3499")
print(f"Accuracy drop: {(0.3499 - miou):.4f} ({(0.3499-miou)/0.3499*100:.1f}%)")