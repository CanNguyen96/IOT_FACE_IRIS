# Multimodal Biometric Authentication System
## Face + Iris Recognition with ResNet18

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hệ thống xác thực sinh trắc học đa phương thức kết hợp nhận dạng khuôn mặt và mống mắt, đạt **99.95% fusion accuracy**.

---

## 🎯 Highlights

- ✅ **Face Recognition**: ResNet18 trained on LFW Funneled, Val Acc 51.77%
- ✅ **Iris Recognition**: ResNet18 trained on CASIA-Iris-Thousand, ~85-90% accuracy
- ✅ **Multimodal Fusion**: 0.3×Face + 0.7×Iris, AUC=1.0000, EER=0.10%, **Accuracy 99.95%**
- ✅ **Real-time Demo**: Live authentication với camera
- ✅ **Production Ready**: Fully trained models sẵn sàng deploy

---

## 📊 Performance

| Model | Dataset | Training Acc | Val Acc | AUC | EER |
|-------|---------|--------------|---------|-----|-----|
| **Face ResNet18** | LFW Funneled (1,680 classes) | 99.99% | 51.77% | - | - |
| **Iris ResNet18** | CASIA-Iris-Thousand (1,000 classes) | 100% | ~85-90% | - | - |
| **Fusion (0.3F + 0.7I)** | Face + Iris | - | - | 1.0000 | 0.10% |

**Fusion Performance:**
- **Accuracy**: 99.95%
- **FAR**: 0.03%
- **FRR**: 0.07%

---

## 🗂️ Project Structure
IOT_FACE_IRIS/
├── face_cnn/                      # Face Recognition Module
│   ├── train_lfw_funneled.py      # Training script
│   ├── generate_embeddings_resnet18.py
│   ├── face_model_improved.py     # ResNet18 architecture
│   ├── face_lfw_funneled_best.pth # Best model (epoch 15)
│   └── face_embeddings_resnet18.npz
│
├── iris_cnn/                      # Iris Recognition Module
│   ├── train_iris_improved.py     # Training script
│   ├── generate_embeddings_resnet18.py
│   ├── iris_model_improved.py     # ResNet18 architecture
│   ├── iris_dataset.py            # Dataset loader
│   ├── iris_cnn_resnet18.pth      # Best model
│   └── iris_embeddings_resnet18.npz
│
├── realtime_demo/                 # Real-time Authentication
│   ├── simple_mfa_demo.py         # MFA demo (face + iris)
│   ├── enroll_yourself.py         # User enrollment
│   └── enrolled_user.npz          # Enrolled user data
│
├── final_fusion_evaluation.py     # Complete system evaluation
├── final_fusion_resnet18_resnet18.png # Fusion ROC curve
├── README.md                      # This file
└── DEPLOYMENT_GUIDE.md            # Deployment instructions
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
pip install opencv-python 
pip install scikit-learn matplotlib tqdm
pip install numpy pandas
pip install mediapipe==0.10.9
```

---

## 📥 Download Dataset & Models

### **Datasets** (Required for Training)

#### 1. **CASIA-Iris-Thousand** (Iris Recognition)
- **Size**: ~2.5 GB
- **Download**: [Kaggle - CASIA-Iris-Thousand](https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand)
- **Extract to**: `CASIA-Iris-Thousand/`
-Funneled (Labeled Faces in the Wild)** (Face Recognition)
- **Size**: ~200 MB
- **Download**: [Official LFW-Funneled](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)
- **Extract to**: `lfw_funneled/`
- **Note**: Filtered to 1,680 classes with ≥2 images per personple](https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data) or [Official LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)
- **Extract to**: `lfw_funneIncluded in repository)

✅ Models are already trained and included:
- `face_cnn/face_lfw_funneled_best.pth` - Face model (epoch 15, val_acc 51.77%)
- `iris_cnn/iris_cnn_resnet18.pth` - Iris model (1000 classes)
- Embeddings files are also pre-generatedet18.pth` (46 MB) - [Download](https://drive.google.com/file/d/1TMtRcGJxoV-eP-MHxAzBLv61sl2Cxdfe/view?usp=sharing)
- **Embeddings**: `iris_embeddings_resnet18.npz` (39 MB) - [Download](https://drive.google.com/file/d/1w4qAhpYn7LuMlr7fxwyIG-hjJeVXriX-/view?usp=sharing)
- **Place in**: `iris_cnn/`

---

## 🚀 Quick Start

### Option 1: Use Pre-trained Models (Recommended)

1. Download models from Google Drive (see above)
2. Run real-time demo:
```bash
cd realtime_demo
#### Step 1: Enroll yourself
```bash
python realtime_demo/enroll_yourself.py
```
- Press **SPACE** to capture your face
- Press **SPACE** to capture your iris
- Creates `enrolled_user.npz` with your embeddings

#### Step 2: Run MFA demo
```bash
python realtime_demo/simple_mfa_demo.py
```
- Press **F** to verify face
- Press **I** to verify iris
- When both verified → **ACCESS GRANTED**
#### Train Face Recognition
```bash
python face_cnn/train_lfw_funneled.py
python face_cnn/generate_embeddings_resnet18.py
```

#### Train Iris Recognition
```bash
python iris_cnn/train_iris_improved.py
python iris_cnn/generate_embeddings_resnet18.py
```

#### Evaluate Fusion
```bash
python final_fusion_evaluation.py
```

---

## 🎥 Real-time Demo

### Quick Demo (Enrollment + Authentication)
```bash
cd realtime_demo
python quick_demo.py
```

### Step 1: Enroll yourself
```bash
python realtime_demo/enroll_yourself.py
```

### Step 2: Run MFA authentication
```bash
python realtime_demo/simple_mfa_demo.py
```

**Controls:**
- **F** - Verify face
- **I** - Verify iris  
- **ESC** - Exit Face Recognition (ResNet18 vs InsightFace)
| Metric | Face ResN
- **Model**: ResNet18 (512-dim embeddings, Softmax loss)
- **Dataset**: LFW Funneled - 1,680 classes, 9,164 training images
- **Training Accuracy**: 99.99%
- **Validation Accuracy**: 51.77% (epoch 15)
- **Note**: Low validation accuracy due to small dataset (avg 5.5 images/person)

### Iris Recognition
- **Model**: ResNet18 (512-dim embeddings)
- **Dataset**: CASIA-Iris-Thousand - 1,000 classes, 20,000 images
- **Training Accuracy**: 100%
- **Validation Accuracy**: ~85-90%

### Multimodal Fusion (Face + Iris)
- **Fusion Strategy**: 0.3 × Face + 0.7 × Iris (score-level)
- **Genuine pairs**: 3,000
- **Impostor pairs**: 3,000
- **AUC**: 1.0000 (Perfect!)
- **EER**: 0.10%
- **Accuracy**: 99.95%
- **FAR**: 0.03%
- **FRR**: 0.07%

**Key Insight**: Despite face CNN having only 51% validation accuracy, fusion with iris achieves 99.95% accuracy!
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
