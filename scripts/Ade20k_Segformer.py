!pip install transformers -q

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from PIL import Image
import numpy as np

class ADE20KDataset(Dataset):
    def __init__(self, root, split='training'):
        self.img_dir  = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'annotations', split)
        self.images   = sorted([
            f for f in os.listdir(self.img_dir) if f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image    = Image.open(img_path).convert('RGB')

        mask_name = self.images[idx].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask      = Image.open(mask_path).resize((512, 512), Image.NEAREST)
        mask      = torch.from_numpy(np.array(mask)).long()
        mask      = mask.clamp(0, 149)

        return image, mask

def collate_fn(batch):
    images = [item[0] for item in batch]  # list of PIL images
    masks  = torch.stack([item[1] for item in batch])  # stack mask tensors
    return images, masks
processor = AutoImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=150,
    ignore_mismatched_sizes=True
)

ROOT   = '/kaggle/input/datasets/awsaf49/ade20k-dataset/ADEChallengeData2016'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

train_dataset = ADE20KDataset(root=ROOT, split='training')  
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

model     = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

for epoch in range(10):
    model.train()
    epoch_loss  = 0
    num_batches = 0

    for i, (images, masks) in enumerate(train_loader):
        inputs       = processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        masks        = masks.to(device)

        outputs = model(pixel_values=pixel_values, labels=masks)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

        if i % 100 == 0:
            print(f"Epoch {epoch+1} batch {i}/{len(train_loader)} — loss: {loss.item():.4f}", flush=True)

    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}", flush=True)
    torch.save(model.state_dict(), f"/kaggle/working/segformer_epoch{epoch+1}.pt")
    print(f"Checkpoint saved: epoch {epoch+1}", flush=True)

print("Training complete!", flush=True)
