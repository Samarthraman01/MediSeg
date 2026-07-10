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
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# ── U-Net ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=150):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

ROOT = os.path.expanduser('~/mediseg/data/ADEChallengeData2016')

train_dataset = ADE20KDataset(root=ROOT, split='training')
val_dataset   = ADE20KDataset(root=ROOT, split='validation')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False)

model     = UNet(in_channels=3, num_classes=150).to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Training on {len(train_dataset)} images")
print(f"Validating on {len(val_dataset)} images")

for epoch in range(20):     # 20 epochs — train overnight
    model.train()

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks  = masks.to(device)

        predictions = model(images)
        loss        = loss_fn(predictions, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch+1} batch {i}/{len(train_loader)} — loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} complete — loss: {loss.item():.4f}")

# save model
torch.save(model.state_dict(), "models/unet_ade20k.pt")
print("Model saved!")