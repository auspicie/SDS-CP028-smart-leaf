import os
import splitfolders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define augmentations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Split dataset
if not os.path.exists("split_dataset"):
    splitfolders.ratio("temp_dataset", output="split_dataset", seed=42, ratio=(0.8, 0.2), group_prefix=None, move=False)

# Load datasets
train_dataset = datasets.ImageFolder("split_dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("split_dataset/val", transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
