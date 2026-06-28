import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(
    root = '~/mediseg/data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_dataset = torchvision.datasets.MNIST(
    root = '~/mediseg/data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle = True 
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size = 32, 
    shuffle = False
)




class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,128)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
model = DigitNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for images, labels in train_loader:
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} - loss: {loss.item():.4f}")

correct = 0
total = 0

with torch.no_grad():   # no gradients needed for testing
    for images, labels in test_loader:
        predictions = model(images)
        predicted_classes = torch.argmax(predictions, dim=1)
        correct += (predicted_classes == labels).sum().item()
        total   += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest accuracy: {accuracy:.2f}%")
print(f"Correctly identified {correct} out of {total} digits")