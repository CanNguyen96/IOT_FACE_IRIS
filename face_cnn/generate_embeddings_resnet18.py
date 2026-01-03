"""
Generate face embeddings using trained ResNet18 model
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from face_model_improved import FaceRecognitionModel
from tqdm import tqdm


class SimpleImageDataset(Dataset):
    """Simple dataset for loading images"""
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load all images
        self.images = []
        self.labels = []
        
        person_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for label, person_dir in enumerate(person_dirs):
            for img_path in person_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]

print("="*70)
print("GENERATE FACE EMBEDDINGS - ResNet18")
print("="*70)

# Configuration
model_path = "face_cnn/face_lfw_funneled_best.pth"
dataset_path = r"D:\data train\dataset\lfw_funneled"
output_path = "face_cnn/face_embeddings_resnet18.npz"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load checkpoint
print(f"\n[INFO] Loading model from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Embedding size: {checkpoint['embedding_size']}")
print(f"  Num classes: {checkpoint['num_classes']}")
print(f"  Val accuracy: {checkpoint['val_acc']:.2f}%")

# Load model
model = FaceRecognitionModel(
    num_classes=checkpoint['num_classes'],
    embedding_size=checkpoint['embedding_size'],
    dropout=checkpoint.get('dropout', 0.5),
    pretrained=False
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("[OK] Model loaded")

# Load dataset
print(f"\n[INFO] Loading dataset from {dataset_path}...")
dataset = SimpleImageDataset(dataset_path)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
print(f"[OK] Dataset loaded: {len(dataset)} samples")

# Extract embeddings
print("\n[INFO] Extracting embeddings...")
all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc='Extracting'):
        imgs = imgs.to(device)
        embeddings = model(imgs, return_embedding=True)  # Get embeddings, not logits
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
