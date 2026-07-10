import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os 
import numpy as np 
import matplotlib.pyplot as plt

class ADE20KDataset(Dataset):
    def __init__(self, root, split='training'):
        self.img_dir = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'annotations', split)

        self.images = sorted(os.listdir(self.img_dir))

        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mask_transform = transforms.Resize(
            (512, 512), 
            interpolation = transforms.InterpolationMode.NEAREST
        )
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # build full image path
        img_path = os.path.join(self.img_dir, self.images[idx])
        # load image
        image = Image.open(img_path).convert('RGB')
        #apply trransform
        image = self.image_transform(image)
        # build mask path — same name but .png instead of .jpg
        mask_name = self.images[idx].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        #load mask
        mask = Image.open(mask_path)
        #apply mask transform
        mask = self.mask_transform(mask)
        #convert to tensor - keeps calss indices  0-150
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# ── Step 3 — DataLoader ───────────────────────────────────────────────
ROOT = os.path.expanduser('~/mediseg/data/ADEChallengeData2016')

train_dataset = ADE20KDataset(root=ROOT, split='training')
val_dataset   = ADE20KDataset(root=ROOT, split='validation')

train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=4, shuffle=False)

# ── Quick test — verify everything loads correctly ─────────────────────
print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

# load one batch and check shapes
images, masks = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Mask batch shape:  {masks.shape}")
print(f"Mask unique values: {masks.unique()}")