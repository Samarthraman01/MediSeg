import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#load CIFAR-10#6000colours images 32*32 ,m1 classes 
#we do this step so that the stored image is in value from 0 to 255 adn then we have to 
#normalize them hence we use it.
data_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#to load the data sets (training dataset)
train_dataset = torchvision.datasets.CIFAR10(
    root ='/Users/samarthramanghanate/mediseg/data', 
    train = True, 
    download = True, 
    transform = data_transform
)

#to load test data sets
test_dataset = torchvision.datasets.CIFAR10(
    root ='/Users/samarthramanghanate/mediseg/data', 
    train = False, 
    download = True, 
    transform = data_transform
)

#data_loader for neural_network so that the output is not biased and thus for better speed we use 32 bacthes too 
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

#data loader
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle = False
)

#nn.Conv2d(in_channels, out_channels, kernel_size)
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convo1=nn.Conv2d(3,32, kernel_size=3, padding=1)
        self.convo2=nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)        # shrinks image by half
        self.fc1  = nn.Linear(8*8*64, 512)   # first fully connected
        self.fc2  = nn.Linear(512, 10)       # output — 10 classes