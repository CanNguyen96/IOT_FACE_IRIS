"""
Training script for face recognition using LFW Funneled dataset with ResNet18 + ArcFace
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_cnn.face_model_improved import FaceRecognitionModel


class LFWFunneledDataset(Dataset):
    """Dataset class for LFW Funneled images"""
    
    def __init__(self, root_dir, transform=None, min_images_per_person=2):
        """
        Args:
            root_dir: Path to lfw_funneled directory
            transform: Optional transform to be applied on images
            min_images_per_person: Minimum number of images per person to include
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all images and create label mapping
        self.images = []
        self.labels = []
        self.label_to_name = {}
        self.name_to_label = {}
        
        # Get all person directories
        person_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        current_label = 0
        for person_dir in person_dirs:
            person_name = person_dir.name
            images = list(person_dir.glob('*.jpg'))
            
            # Only include persons with minimum number of images
            if len(images) >= min_images_per_person:
                self.label_to_name[current_label] = person_name
                self.name_to_label[person_name] = current_label
                
                for img_path in images:
                    self.images.append(img_path)
                    self.labels.append(current_label)
                
                current_label += 1
        
        self.num_classes = current_label
        print(f"Loaded {len(self.images)} images from {self.num_classes} people")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(data_dir, batch_size=32, num_workers=4):
    """Create train and validation data loaders"""
    
    # Data augmentation for training - STRONGER to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added
        transforms.RandomGrayscale(p=0.1),  # Added
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # Added
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load full dataset
    full_dataset = LFWFunneledDataset(data_dir, transform=train_transform, min_images_per_person=2)
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.num_classes


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), 100 * correct / total


def main():
    # Configuration
    DATA_DIR = r"d:\data train\dataset\lfw_funneled"
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0005  # Reduced from 0.001
    EMBEDDING_SIZE = 512
    DROPOUT = 0.5  # Dropout rate
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, num_classes = create_data_loaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE,
        num_workers=0  # Windows compatibility
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Create model with softmax classifier
    print("Creating model...")
    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_size=EMBEDDING_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Use CrossEntropyLoss (simpler and less prone to overfitting)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler - reduce more frequently
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.6)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'embedding_size': EMBEDDING_SIZE,
                'dropout': DROPOUT
            }, 'face_cnn/face_lfw_funneled_best.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'embedding_size': EMBEDDING_SIZE,
                'dropout': DROPOUT
            }, f'face_cnn/face_lfw_funneled_epoch_{epoch}.pth')
    
    # Save final model
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'num_classes': num_classes,
        'embedding_size': EMBEDDING_SIZE,
        'dropout': DROPOUT
    }, 'face_cnn/face_lfw_funneled_final.pth')
    
    # Save history
    with open('face_cnn/face_lfw_funneled_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
