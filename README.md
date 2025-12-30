# Multimodal Biometric Authentication System
## Face + Iris Recognition with ResNet18

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hệ thống xác thực sinh trắc học đa phương thức kết hợp nhận dạng khuôn mặt và mống mắt, đạt **100% accuracy** trên cả hai modality.

---

## 🎯 Highlights

- ✅ **Face Recognition**: ResNet18 custom-trained, AUC=1.0000, EER=0.00%
- ✅ **Iris Recognition**: ResNet18 custom-trained, AUC=1.0000, EER=0.03%
- ✅ **Multimodal Fusion**: Score-level fusion, AUC=1.0000, EER=0.00%
- ✅ **Real-time Demo**: Live authentication với camera
- ✅ **Production Ready**: Fully trained models sẵn sàng deploy

---

## 📊 Performance

| Model | Dataset | Classes | Samples | AUC | EER | Accuracy |
|-------|---------|---------|---------|-----|-----|----------|
| **Face ResNet18** | LFW | 5,749 | 13,233 | 1.0000 | 0.00% | 100% |
| **Iris ResNet18** | CASIA | 1,000 | 20,000 | 1.0000 | 0.03% | 100% |
| **Fusion (0.3F + 0.7I)** | Both | 1,000 | 6,000 pairs | 1.0000 | 0.00% | 100% |

**Comparison với InsightFace pretrained:**
- Face ResNet18: **+46.4% AUC**, **-100% EER** (tốt hơn rất nhiều!)

---

## 🗂️ Project Structure

```
dataset/
├── face_cnn/                      # Face Recognition Module
│   ├── train_face_resnet18.py     # Training script
│   ├── generate_embeddings_resnet18.py
│   ├── evaluate_face_resnet18.py
│   ├── face_model_improved.py     # ResNet18 architecture
│   ├── face_utils.py              # Utilities
│   └── face_dataset.py            # LFW dataset loader
│
├── iris_cnn/                      # Iris Recognition Module
│   ├── train_iris_improved.py     # Training script
│   ├── generate_embeddings_resnet18.py
│   ├── evaluate_resnet18.py
│   ├── fusion_evaluate_resnet18.py
│   └── iris_model_improved.py     # ResNet18 architecture
│
├── realtime_demo/                 # Real-time Authentication
│   ├── quick_demo.py              # Interactive demo
│   ├── mfa_realtime.py            # Continuous authentication
│   ├── prepare_enrollment.py      # User enrollment
│   └── iris_segmentation.py       # MediaPipe iris detection
│
├── final_fusion_evaluation.py     # Complete system evaluation
└── IMPROVEMENT_REPORT.md          # Detailed performance report
```

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/CanNguyen96/IOT_FACE_IRIS.git
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python mediapipe
pip install scikit-learn matplotlib tqdm
pip install numpy pandas
```

---

## 📥 Download Dataset & Models

### **Datasets** (Required for Training)

#### 1. **CASIA-Iris-Thousand** (Iris Recognition)
- **Size**: ~2.5 GB
- **Download**: [Kaggle - CASIA-Iris-Thousand](https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand)
- **Extract to**: `CASIA-Iris-Thousand/`

#### 2. **LFW (Labeled Faces in the Wild)** (Face Recognition)
- **Size**: ~200 MB
- **Download**: [Kaggle - LFW People](https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data) or [Official LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)
- **Extract to**: `lfw_funneled/`

### **Pre-trained Models** (Optional - Skip Training)

#### Face ResNet18
- **Model**: `face_cnn_resnet18.pth` (55 MB) - [Download](https://drive.google.com/file/d/117uxjdg2-bax0Q1nsXhFHFFHR_udDuvS/view?usp=sharing)
- **Embeddings**: `face_embeddings_resnet18.npz` (27 MB) - [Download](https://drive.google.com/file/d/18Bd616tpcUVNXkKiQ2ShVeQG3wcEhjQV/view?usp=sharing)
- **Place in**: `face_cnn/`

#### Iris ResNet18
- **Model**: `iris_cnn_resnet18.pth` (46 MB) - [Download](https://drive.google.com/file/d/1TMtRcGJxoV-eP-MHxAzBLv61sl2Cxdfe/view?usp=sharing)
- **Embeddings**: `iris_embeddings_resnet18.npz` (39 MB) - [Download](https://drive.google.com/file/d/1w4qAhpYn7LuMlr7fxwyIG-hjJeVXriX-/view?usp=sharing)
- **Place in**: `iris_cnn/`

---

## 🚀 Quick Start

### Option 1: Use Pre-trained Models (Recommended)

1. Download models from Google Drive (see above)
2. Run real-time demo:
```bash
cd realtime_demo
python quick_demo.py
```

**Demo Instructions:**
- Press **SPACE** to enroll your face + iris
- Press **SPACE** again to authenticate
- Press **ESC** to exit

### Option 2: Train From Scratch

#### Train Face Recognition
```bash
cd face_cnn
python train_face_resnet18.py --epochs 30 --batch_size 32 --lr 0.0001
python generate_embeddings_resnet18.py
python evaluate_face_resnet18.py
```

#### Train Iris Recognition
```bash
cd iris_cnn
python train_iris_improved.py --epochs 30 --batch_size 64 --lr 0.001
python generate_embeddings_resnet18.py
python evaluate_resnet18.py
```

#### Evaluate Fusion
```bash
python fusion_evaluate_resnet18.py
cd ..
python final_fusion_evaluation.py
```

---

## 🎥 Real-time Demo

### Quick Demo (Enrollment + Authentication)
```bash
cd realtime_demo
python quick_demo.py
```

### MFA Real-time (Continuous Verification)
```bash
# Step 1: Enroll user
python prepare_enrollment.py

# Step 2: Run authentication
python mfa_realtime.py
```

---

## 📈 Evaluation Results

### Face Recognition (ResNet18 vs InsightFace)
| Metric | Face ResNet18 | InsightFace | Improvement |
|--------|---------------|-------------|-------------|
| AUC | **1.0000** | 0.6832 | +46.4% |
| EER | **0.00%** | 37.30% | -100% |

### Iris Recognition (ResNet18 vs Simple CNN)
| Metric | Iris ResNet18 | Simple CNN | Improvement |
|--------|---------------|------------|-------------|
| AUC | **1.0000** | 0.8500 | +17.6% |
| EER | **0.03%** | 19.10% | -99.8% |

### Final Multimodal Fusion
- **Fusion Strategy**: 0.3 × Face + 0.7 × Iris
- **AUC**: 1.0000 (Perfect!)
- **EER**: 0.00%
- **Accuracy**: 100%

---

## 🛠️ Technical Details

### Architecture
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Face Embedding**: 512-D L2-normalized vectors
- **Iris Embedding**: 512-D L2-normalized vectors
- **Similarity Metric**: Cosine Similarity
- **Fusion Method**: Score-level weighted sum

### Training
- **Optimizer**: Adam
- **Learning Rate**: 0.0001 (Face), 0.001 (Iris)
- **Scheduler**: StepLR (gamma=0.5, step=10)
- **Epochs**: 30
- **Device**: CPU/CUDA

### Iris Segmentation
- **Method**: MediaPipe Face Mesh
- **Landmarks**: 468-477 (iris contour)
- **Size**: 512×64 grayscale
- **Normalization**: MinMax scaling

---

## 🙏 Acknowledgments

- **Datasets**: 
  - [LFW](http://vis-www.cs.umass.edu/lfw/) - Face Recognition
  - [CASIA-Iris-Thousand](http://biometrics.idealtest.org/) - Iris Recognition
- **Libraries**: PyTorch, MediaPipe, OpenCV, scikit-learn

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: canhayqua012@gmail.com
- GitHub: https://github.com/CanNguyen96

---

**⭐ Star this repo if you find it helpful!**
