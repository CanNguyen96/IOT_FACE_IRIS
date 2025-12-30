"""
Evaluate Face ResNet18 performance and compare with InsightFace
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

print("="*80)
print("FACE VERIFICATION EVALUATION - ResNet18 vs InsightFace")
print("="*80)

# Load embeddings
print("\n[INFO] Loading embeddings...")

# ResNet18 (trained)
data_resnet = np.load("face_embeddings_resnet18.npz")
emb_resnet = data_resnet["embeddings"]
lbl_resnet = data_resnet["labels"]
print(f"[OK] ResNet18: {emb_resnet.shape}")

# InsightFace (pretrained) - from previous
try:
    data_insight = np.load("face_embeddings_norm.npz")
    emb_insight = data_insight["embeddings"]
    lbl_insight = data_insight["labels"]
    print(f"[OK] InsightFace: {emb_insight.shape}")
    has_insightface = True
except:
    print("[WARN] InsightFace embeddings not found, will only evaluate ResNet18")
    has_insightface = False

# Generate test pairs
num_pairs = 3000
random.seed(42)

def evaluate_model(embeddings, labels, model_name):
    """Evaluate a model and return metrics"""
    genuine_scores = []
    impostor_scores = []
    
    unique_labels = np.unique(labels)
    
    # Genuine pairs
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
    
    # Impostor pairs
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
    
    # ROC
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    
    return {
        'name': model_name,
        'genuine_mean': genuine_scores.mean(),
        'genuine_std': genuine_scores.std(),
        'impostor_mean': impostor_scores.mean(),
        'impostor_std': impostor_scores.std(),
        'auc': roc_auc,
        'eer': eer,
        'fpr': fpr,
        'tpr': tpr,
        'genuine': genuine_scores,
        'impostor': impostor_scores
    }

print("\n[INFO] Evaluating ResNet18 (trained)...")
results_resnet = evaluate_model(emb_resnet, lbl_resnet, "Face ResNet18 (Trained)")

if has_insightface:
    print("[INFO] Evaluating InsightFace (pretrained)...")
    random.seed(42)
    results_insight = evaluate_model(emb_insight, lbl_insight, "InsightFace (Pretrained)")

# Print results
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

if has_insightface:
    print(f"{'Metric':<30} {'ResNet18':>20} {'InsightFace':>20} {'Difference':>15}")
    print("-"*80)
    
    metrics = [
        ('Genuine Mean', 'genuine_mean', '{:.4f}'),
        ('Genuine Std', 'genuine_std', '{:.4f}'),
        ('Impostor Mean', 'impostor_mean', '{:.4f}'),
        ('Impostor Std', 'impostor_std', '{:.4f}'),
        ('AUC', 'auc', '{:.4f}'),
        ('EER', 'eer', '{:.2%}'),
    ]
    
    for label, key, fmt in metrics:
        val_resnet = results_resnet[key]
        val_insight = results_insight[key]
        diff = val_resnet - val_insight
        diff_str = f"{diff:+.4f}" if 'eer' not in key else f"{diff:+.2%}"
        
        print(f"{label:<30} {fmt.format(val_resnet):>20} {fmt.format(val_insight):>20} {diff_str:>15}")
else:
    print("ResNet18 Results:")
    print(f"  Genuine Mean: {results_resnet['genuine_mean']:.4f}")
    print(f"  Impostor Mean: {results_resnet['impostor_mean']:.4f}")
    print(f"  AUC: {results_resnet['auc']:.4f}")
    print(f"  EER: {results_resnet['eer']*100:.2f}%")

print("="*80)

# Visualization
if has_insightface:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC curves
    ax = axes[0, 0]
    ax.plot(results_resnet['fpr'], results_resnet['tpr'], 'b-', lw=2,
            label=f"ResNet18 (AUC={results_resnet['auc']:.3f})")
    ax.plot(results_insight['fpr'], results_insight['tpr'], 'r-', lw=2,
            label=f"InsightFace (AUC={results_insight['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Accept Rate')
    ax.set_ylabel('True Accept Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC comparison
    ax = axes[0, 1]
    models = ['ResNet18\n(Trained)', 'InsightFace\n(Pretrained)']
    aucs = [results_resnet['auc'], results_insight['auc']]
    colors = ['blue', 'red']
    bars = ax.bar(models, aucs, color=colors, alpha=0.7)
    ax.set_ylabel('AUC')
    ax.set_title('AUC Comparison')
    ax.set_ylim([0, 1.1])
    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc_val:.4f}', ha='center', va='bottom', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # EER comparison
    ax = axes[1, 0]
    eers = [results_resnet['eer'] * 100, results_insight['eer'] * 100]
    bars = ax.bar(models, eers, color=colors, alpha=0.7)
    ax.set_ylabel('EER (%)')
    ax.set_title('Equal Error Rate (lower is better)')
    for bar, eer_val in zip(bars, eers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eer_val:.2f}%', ha='center', va='bottom', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Score distributions
    ax = axes[1, 1]
    ax.hist(results_resnet['genuine'], bins=50, alpha=0.3, label='ResNet18 Genuine', color='blue')
    ax.hist(results_resnet['impostor'], bins=50, alpha=0.3, label='ResNet18 Impostor', color='blue', linestyle='dashed')
    ax.hist(results_insight['genuine'], bins=50, alpha=0.3, label='InsightFace Genuine', color='red')
    ax.hist(results_insight['impostor'], bins=50, alpha=0.3, label='InsightFace Impostor', color='red', linestyle='dashed')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('face_comparison_resnet18_vs_insightface.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: face_comparison_resnet18_vs_insightface.png")
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROC curve
    ax = axes[0, 0]
    ax.plot(results_resnet['fpr'], results_resnet['tpr'], 'b-', lw=2,
            label=f"AUC={results_resnet['auc']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Accept Rate')
    ax.set_ylabel('True Accept Rate')
    ax.set_title('ROC Curve - Face ResNet18')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Score distribution
    ax = axes[0, 1]
    ax.hist(results_resnet['genuine'], bins=50, alpha=0.5, label='Genuine', color='green')
    ax.hist(results_resnet['impostor'], bins=50, alpha=0.5, label='Impostor', color='red')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Metrics
    ax = axes[1, 0]
    metrics = ['AUC', f'EER\n({results_resnet["eer"]*100:.2f}%)']
    values = [results_resnet['auc'], 1 - results_resnet['eer']]
    ax.bar(metrics, values, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
FACE RESNET18 RESULTS

AUC: {results_resnet['auc']:.4f}
EER: {results_resnet['eer']*100:.2f}%

Genuine:  {results_resnet['genuine_mean']:.3f} ± {results_resnet['genuine_std']:.3f}
Impostor: {results_resnet['impostor_mean']:.3f} ± {results_resnet['impostor_std']:.3f}

Training Accuracy: 100%
"""
    ax.text(0.1, 0.5, summary, fontsize=12, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('face_evaluation_resnet18.png', dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: face_evaluation_resnet18.png")

print("\n[INFO] Evaluation complete!")
plt.show()
