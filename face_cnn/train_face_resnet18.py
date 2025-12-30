"""
Train Face CNN with ResNet18 backbone on LFW dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from face_dataset import FaceDataset
import argparse
from tqdm import tqdm
import os
import sys

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_model_improved import FaceCNN_ResNet

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
    parser = argparse.ArgumentParser(description='Train Face CNN on LFW')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18'],
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
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--dataset_path', type=str,
                        default=r"D:\data train\dataset\lfw_funneled",
                        help='Path to LFW dataset')
    parser.add_argument('--output', type=str, default='face_cnn_resnet18.pth',
                        help='Output model filename')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FACE CNN TRAINING - LFW DATASET")
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
    print("\n[INFO] Loading LFW dataset...")
    dataset = FaceDataset(args.dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    num_classes = len(set([label for _, label in dataset.samples]))
    print(f"[OK] Dataset loaded: {len(dataset)} samples, {num_classes} classes")
    
    # Create model
    print(f"\n[INFO] Creating ResNet18 model...")
    model = FaceCNN_ResNet(
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
