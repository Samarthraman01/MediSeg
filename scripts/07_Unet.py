import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
import numpy as np

# ── Transforms ────────────────────────────────────────────────────────
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256),
        interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# ── Dataset ───────────────────────────────────────────────────────────
class PetsDataset(Dataset):
    def __init__(self, split='trainval'):
        self.dataset = OxfordIIITPet(
            root='~/mediseg/data',
            split=split,
            target_types='segmentation',
            download=True
        )
        self.image_transform = image_transform
        self.mask_transform  = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = self.image_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask * 255).long()
        mask  = (mask == 1).long()
        mask  = mask.squeeze(0)
        return image, mask

# ── ConvBlock ─────────────────────────────────────────────────────────
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
    def __init__(self, in_channels=3, num_classes=2):
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

# ── Setup ─────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

train_data   = PetsDataset(split='trainval')
test_data    = PetsDataset(split='test')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=8, shuffle=True
)
test_loader  = torch.utils.data.DataLoader(
    test_data, batch_size=8, shuffle=False
)

model     = UNet(in_channels=3, num_classes=2).to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Training loop ─────────────────────────────────────────────────────
for epoch in range(10):
    model.train()
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks  = masks.to(device)

        predictions = model(images)
        loss        = loss_fn(predictions, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Epoch {epoch+1} batch {i}/{len(train_loader)} — loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} complete — loss: {loss.item():.4f}")

torch.save(model.state_dict(), "models/unet_pets.pt")
print("Model saved!")