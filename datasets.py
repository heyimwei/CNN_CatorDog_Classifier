import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定義 transform
train_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# 載入資料
train_ds = datasets.ImageFolder('data/train', transform=train_transforms)
val_ds   = datasets.ImageFolder('data/val',   transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32)

print(f"Train classes: {train_ds.classes}, Samples: {len(train_ds)}")
print(f"Val   classes: {val_ds.classes}, Samples: {len(val_ds)}")
