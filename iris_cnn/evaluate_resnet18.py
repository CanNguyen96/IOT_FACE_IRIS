"""
Evaluate improved ResNet18 model
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

print("="*80)
print("IRIS VERIFICATION EVALUATION - ResNet18 MODEL")
print("="*80)

# Load embeddings
print("\n[INFO] Loading ResNet18 embeddings...")
data = np.load("iris_embeddings_resnet18.npz")
embeddings = data["embeddings"]
labels = data["labels"]

print(f"[OK] Loaded: {embeddings.shape}, {len(np.unique(labels))} classes")

# Generate test pairs
num_pairs = 3000
random.seed(42)

genuine_scores = []
impostor_scores = []

unique_labels = np.unique(labels)

print(f"\n[INFO] Generating {num_pairs} genuine pairs...")
attempts = 0
max_attempts = num_pairs * 10
while len(genuine_scores) < num_pairs and attempts < max_attempts:
    attempts += 1
    label = random.choice(unique_labels)
    idx = np.where(labels == label)[0]
    if len(idx) < 2:
        continue
    i1, i2 = random.sample(list(idx), 2)
    score = cosine_similarity(
        embeddings[i1].reshape(1, -1),
        embeddings[i2].reshape(1, -1)
    )[0][0]
    genuine_scores.append(score)

print(f"[INFO] Generating {num_pairs} impostor pairs...")
for _ in range(num_pairs):
    l1, l2 = random.sample(list(unique_labels), 2)
    i1 = random.choice(np.where(labels == l1)[0])
    i2 = random.choice(np.where(labels == l2)[0])
    score = cosine_similarity(
        embeddings[i1].reshape(1, -1),
        embeddings[i2].reshape(1, -1)
    )[0][0]
    impostor_scores.append(score)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"[OK] Generated {len(genuine_scores)} genuine and {len(impostor_scores)} impostor pairs")

# Calculate metrics
print("\n" + "="*80)
print("SCORE DISTRIBUTION")
print("="*80)
print(f"Genuine mean    : {genuine_scores.mean():.4f}")
print(f"Genuine std     : {genuine_scores.std():.4f}")
print(f"Impostor mean   : {impostor_scores.mean():.4f}")
print(f"Impostor std    : {impostor_scores.std():.4f}")

# ROC curve
y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
y_scores = np.concatenate([genuine_scores, impostor_scores])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# EER
fnr = 1 - tpr
eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
eer = fpr[eer_idx]
eer_threshold = thresholds[eer_idx]

print("\n" + "="*80)
print("ROC ANALYSIS")
print("="*80)
print(f"AUC (Area Under Curve) : {roc_auc:.4f}")
print(f"EER (Equal Error Rate) : {eer*100:.2f}%")
print(f"EER Threshold          : {eer_threshold:.4f}")

# Performance at optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
far = fpr[optimal_idx]
frr = fnr[optimal_idx]

print("\n" + "="*80)
print(f"PERFORMANCE AT THRESHOLD = {optimal_threshold:.4f}")
print("="*80)
print(f"FAR (False Accept Rate)  : {far*100:.2f}%")
print(f"FRR (False Reject Rate)  : {frr*100:.2f}%")
print(f"Accuracy                 : {(1-far)*(1-frr)*100:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC curve
ax = axes[0, 0]
ax.plot(fpr, tpr, 'b-', lw=2, label=f'ResNet18 (AUC={roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.plot(eer, 1-eer, 'ro', markersize=8, label=f'EER={eer*100:.1f}%')
ax.set_xlabel('False Accept Rate')
ax.set_ylabel('True Accept Rate')
ax.set_title('ROC Curve - ResNet18')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Score distribution
ax = axes[0, 1]
ax.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='green')
ax.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red')
ax.axvline(eer_threshold, color='blue', linestyle='--', lw=2, label=f'EER Threshold={eer_threshold:.3f}')
ax.set_xlabel('Similarity Score')
ax.set_ylabel('Frequency')
ax.set_title('Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# FAR vs FRR
ax = axes[1, 0]
ax.plot(thresholds, fpr, 'r-', label='FAR', lw=2)
ax.plot(thresholds, fnr, 'b-', label='FRR', lw=2)
ax.axvline(eer_threshold, color='green', linestyle='--', lw=1, label=f'EER Threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Error Rate')
ax.set_title('FAR vs FRR')
ax.legend()
ax.grid(True, alpha=0.3)

# DET curve
ax = axes[1, 1]
ax.plot(fpr, fnr, 'b-', lw=2)
ax.plot(eer, eer, 'ro', markersize=8, label=f'EER={eer*100:.1f}%')
ax.set_xlabel('False Accept Rate')
ax.set_ylabel('False Reject Rate')
ax.set_title('DET Curve (Detection Error Tradeoff)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('iris_evaluation_resnet18.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved: iris_evaluation_resnet18.png")

plt.show()
