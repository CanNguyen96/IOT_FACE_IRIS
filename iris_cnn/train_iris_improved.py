"""
Training script for improved Iris CNN models
Supports: ResNet18, MobileNet, EfficientNet backbones
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from iris_dataset import IrisDataset
from iris_model_improved import get_iris_model
import argparse
from tqdm import tqdm
import os

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(loader), 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train improved Iris CNN')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['simple', 'resnet18', 'mobilenet', 'efficientnet'],
                        help='Model backbone')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower for pretrained)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--dataset_path', type=str,
                        default=r"D:\data train\dataset\CASIA-Iris-Thousand\CASIA-Iris-Thousand",
                        help='Path to iris dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model filename')
    
    args = parser.parse_args()
    
    # Set default output name
    if args.output is None:
        args.output = f"iris_cnn_{args.backbone}.pth"
    
    print("="*70)
    print("IMPROVED IRIS CNN TRAINING")
    print("="*70)
    print(f"Backbone        : {args.backbone}")
    print(f"Embedding dim   : {args.embedding_dim}")
    print(f"Pretrained      : {args.pretrained}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Device          : {args.device}")
    print("="*70)
    
    # Load dataset
    print("\n[INFO] Loading dataset...")
    dataset = IrisDataset(args.dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    num_classes = len(set([label for _, label in dataset.samples]))
    print(f"[OK] Dataset loaded: {len(dataset)} samples, {num_classes} classes")
    
    # Create model
    print(f"\n[INFO] Creating {args.backbone} model...")
    model = get_iris_model(
        backbone=args.backbone,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created: {total_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print("\n[INFO] Starting training...")
    best_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        loss, acc = train_epoch(model, loader, criterion, optimizer, args.device)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Acc:  {acc:.2f}%")
        print(f"  LR:   {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.output)
            print(f"  [OK] Best model saved: {args.output} (Acc: {acc:.2f}%)")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best accuracy   : {best_acc:.2f}%")
    print(f"Model saved     : {args.output}")
    print("="*70)
    
    # Save metadata
    metadata = {
        'backbone': args.backbone,
        'embedding_dim': args.embedding_dim,
        'num_classes': num_classes,
        'best_acc': best_acc,
        'pretrained': args.pretrained
    }
    torch.save(metadata, args.output.replace('.pth', '_metadata.pth'))
    print(f"[OK] Metadata saved: {args.output.replace('.pth', '_metadata.pth')}")


if __name__ == "__main__":
    main()
