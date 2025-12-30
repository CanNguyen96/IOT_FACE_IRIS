"""
Generate embeddings using improved ResNet18 model
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from iris_dataset import IrisDataset
from iris_model_improved import get_iris_model
from tqdm import tqdm

print("="*70)
print("GENERATE EMBEDDINGS - IMPROVED IRIS CNN (ResNet18)")
print("="*70)

# Configuration
model_path = "iris_cnn_resnet18.pth"
metadata_path = "iris_cnn_resnet18_metadata.pth"
dataset_path = r"D:\data train\dataset\CASIA-Iris-Thousand\CASIA-Iris-Thousand"
output_path = "iris_embeddings_resnet18.npz"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load metadata
print(f"\n[INFO] Loading model metadata...")
metadata = torch.load(metadata_path, weights_only=True)
print(f"  Backbone: {metadata['backbone']}")
print(f"  Embedding dim: {metadata['embedding_dim']}")
print(f"  Num classes: {metadata['num_classes']}")
print(f"  Best accuracy: {metadata['best_acc']:.2f}%")

# Load model
print(f"\n[INFO] Loading model from {model_path}...")
model = get_iris_model(
    backbone=metadata['backbone'],
    num_classes=metadata['num_classes'],
    embedding_dim=metadata['embedding_dim'],
    pretrained=False
).to(device)

model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print("[OK] Model loaded")

# Load dataset
print(f"\n[INFO] Loading dataset from {dataset_path}...")
dataset = IrisDataset(dataset_path)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)  # num_workers=0 for Windows
print(f"[OK] Dataset loaded: {len(dataset)} samples")

# Extract embeddings
print("\n[INFO] Extracting embeddings...")
all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc='Extracting'):
        imgs = imgs.to(device)
        embeddings = model(imgs, return_embedding=True)  # Use forward's return_embedding parameter
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())

# Concatenate
embeddings = np.vstack(all_embeddings)
labels = np.concatenate(all_labels)

# L2 normalize
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print(f"\n[OK] Embeddings extracted: {embeddings.shape}")
print(f"  Labels: {labels.shape}")
print(f"  Unique classes: {len(np.unique(labels))}")

# Save
print(f"\n[INFO] Saving to {output_path}...")
np.savez(output_path, embeddings=embeddings_norm, labels=labels)
print(f"[OK] Saved: {output_path}")

print("\n" + "="*70)
print("EMBEDDINGS GENERATION COMPLETED")
print("="*70)
print(f"Output file: {output_path}")
print(f"Shape: {embeddings_norm.shape}")
print(f"Dtype: {embeddings_norm.dtype}")
print("="*70)
