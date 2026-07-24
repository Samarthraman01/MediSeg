import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

# ── Dataset ───────────────────────────────────────────────────────────
class ADE20KDataset(Dataset):
    def __init__(self, root, split='training'):
        self.img_dir  = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'annotations', split)
        self.images   = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transform = transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image    = Image.open(img_path).convert('RGB')
        image    = self.image_transform(image)

        mask_name = self.images[idx].replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask      = Image.open(mask_path)
        mask      = self.mask_transform(mask)
        mask      = torch.from_numpy(np.array(mask)).long()
        mask      = mask.clamp(0, 149)

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

if __name__ == "__main__":
    # ── Setup ─────────────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}", flush=True)

    ROOT = os.path.expanduser('~/mediseg/data/ADEChallengeData2016')

    train_dataset = ADE20KDataset(root=ROOT, split='training')
    val_dataset   = ADE20KDataset(root=ROOT, split='validation')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=2)

    print(f"Training images:   {len(train_dataset)}", flush=True)
    print(f"Validation images: {len(val_dataset)}", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = UNet(in_channels=3, num_classes=150).to(device)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    # ── Load pretrained weights ───────────────────────────────────────────
    model_path = os.path.expanduser('~/mediseg/models/unet_ade20k_epoch10.pt')

    if os.path.exists(model_path):
        print("Found trained model — skipping training, running evaluation only", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device))

    else:
        print("No trained model found — starting training", flush=True)

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(10):
            model.train()
            epoch_loss  = 0
            num_batches = 0

            for i, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks  = masks.to(device)

                predictions = model(images)
                loss        = loss_fn(predictions, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss  += loss.item()
                num_batches += 1

                if i % 100 == 0:
                    print(f"Epoch {epoch+1} batch {i}/{len(train_loader)} — loss: {loss.item():.4f}", flush=True)

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}", flush=True)
            scheduler.step(avg_loss)

            torch.save(model.state_dict(),
                os.path.expanduser(f'~/mediseg/models/unet_ade20k_epoch{epoch+1}.pt'))
            print(f"Checkpoint saved: epoch {epoch+1}", flush=True)

        print("Training complete!", flush=True)

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\nStarting evaluation...", flush=True)
    model.eval()

    # class names
    class_names = [
        'wall', 'building', 'sky', 'floor', 'tree',
        'ceiling', 'road', 'bed', 'window', 'grass',
        'cabinet', 'sidewalk', 'person', 'earth', 'door',
        'table', 'mountain', 'plant', 'curtain', 'chair'
    ]
    for i in range(20, 150):
        class_names.append(f'class_{i}')

    # mIoU function
    def calculate_miou(model, loader, num_classes, device):
        intersection = torch.zeros(num_classes)
        union        = torch.zeros(num_classes)

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device)
                masks  = masks.to(device)
                outputs   = model(images)
                predicted = torch.argmax(outputs, dim=1)

                for cls in range(num_classes):
                    pred_cls   = (predicted == cls)
                    target_cls = (masks == cls)
                    intersection[cls] += (pred_cls & target_cls).sum().item()
                    union[cls]        += (pred_cls | target_cls).sum().item()

        iou_per_class = intersection / (union + 1e-6)
        miou = iou_per_class.mean().item()
        return miou, iou_per_class

    # run evaluation
    miou, iou_per_class = calculate_miou(model, val_loader, 150, device)

    print(f"\nOverall mIoU: {miou:.4f}", flush=True)

    # sort by IoU
    iou_list   = [(class_names[i], iou_per_class[i].item()) for i in range(150)]
    iou_sorted = sorted(iou_list, key=lambda x: x[1], reverse=True)

    print("\nTop 10 best classes:")
    for name, iou in iou_sorted[:10]:
        print(f"  {name:20s} IoU: {iou:.4f}")

    print("\nBottom 10 worst classes:")
    for name, iou in iou_sorted[-10:]:
        print(f"  {name:20s} IoU: {iou:.4f}")

    # bar chart
    plt.figure(figsize=(20, 6))
    ious   = [x[1] for x in iou_sorted]
    colors = ['green' if iou > 0.3 else 'orange' if iou > 0.1 else 'red' for iou in ious]
    plt.bar(range(150), ious, color=colors)
    plt.axhline(y=miou, color='blue', linestyle='--', label=f'mIoU = {miou:.4f}')
    plt.xlabel('Class (sorted by IoU)')
    plt.ylabel('IoU Score')
    plt.title('U-Net ADE20K — IoU per class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.expanduser('~/mediseg/outputs/ade20k_miou.png'))
    plt.show()
    print("Chart saved to outputs/ade20k_miou.png", flush=True)