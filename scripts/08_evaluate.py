import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, DataLoader


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
    
def calculate_miou(model, loader, num_classes, device):
    model.eval()

    # track intersection and union per class
    intersection = torch.zeros(num_classes)
    union        = torch.zeros(num_classes)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            # get predictions
            outputs  = model(images)
            predicted = torch.argmax(outputs, dim=1)

            # calculate intersection and union for each class
            for cls in range(num_classes):
                pred_cls   = (predicted == cls)
                target_cls = (masks == cls)

                intersection[cls] += (pred_cls & target_cls).sum().item()
                union[cls]        += (pred_cls | target_cls).sum().item()

    # calculate IoU per class
    iou_per_class = intersection / (union + 1e-6)  # +1e-6 avoids divide by zero

    miou = iou_per_class.mean().item()
    return miou, iou_per_class


# ── Setup ─────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = UNet(in_channels=3, num_classes=2).to(device)
model.load_state_dict(torch.load("models/unet_pets.pt", map_location=device))

# test loader — same as training
test_data   = PetsDataset(split='test')
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# calculate
miou, iou_per_class = calculate_miou(model, test_loader, num_classes=2, device=device)

print(f"Background IoU: {iou_per_class[0]:.4f}")
print(f"Pet IoU:        {iou_per_class[1]:.4f}")
print(f"mIoU:           {miou:.4f}")