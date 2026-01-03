# 📤 Deployment Guide - GitHub & Google Drive

## 🗂️ File Organization

### Files to Upload to Google Drive

#### **Large Dataset Files** (~2.7 GB)
```
CASIA-Iris-Thousand/          # 2.5 GB - Iris dataset
lfw_funneled/                 # 200 MB - Face dataset
```

#### **Trained Models** (~150 MB)
```
face_cnn/
├── face_lfw_funneled_best.pth         # 55 MB (epoch 15, val_acc 51.77%)
└── face_embeddings_resnet18.npz       # 27 MB

iris_cnn/
├── iris_cnn_resnet18.pth              # 46 MB
└── iris_embeddings_resnet18.npz       # 39 MB

realtime_demo/
└── enrolled_user.npz                  # <1 MB (user embeddings)
```

#### **Evaluation Results** (Optional - ~5 MB)
```
*.png                          # ROC curves, comparisons
```

---

## 📦 Google Drive Structure

Create folders như sau:

```
Multimodal-Biometric-Models/
├── datasets/
│   ├── CASIA-Iris-Thousand.zip        # Upload compressed
│   └── lfw-funneled.tgz               # Original archive
│
├── face_models/
│   ├── face_lfw_funneled_best.pth
│   └── face_embeddings_resnet18.npz
│
├── iris_models/
│   ├── iris_cnn_resnet18.pth
│   └── iris_embeddings_resnet18.npz
│
└── evaluation_results/
    └── final_fusion_resnet18_resnet18.png
```

### Share Settings
- Set to **"Anyone with the link can view"**
- Copy share links for README.md

---

## 🚀 GitHub Deployment Steps

### 1. Initialize Git Repository
```bash
cd "d:\data train\dataset"
git init
```

### 2. Configure Git
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `multimodal-biometric-auth`
3. Description: "Face + Iris Recognition with ResNet18 - 99.95% Fusion Accuracy"
4. Public/Private: Choose as needed
5. **Do NOT** initialize with README (we already have one)

### 4. Add Remote
```bash
git remote add origin https://github.com/YOUR_USERNAME/multimodal-biometric-auth.git
```

### 5. Stage Files
```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

### 6. Commit
```bash
git commit -m "Initial commit: Multimodal biometric system with ResNet18

- Face Recognition: ResNet18 on LFW Funneled (val_acc 51.77%)
- Iris Recognition: ResNet18 on CASIA-Iris-Thousand (~85-90% acc)
- Multimodal Fusion: 99.95% accuracy (FAR 0.03%, EER 0.10%)
- Real-time MFA demo with webcam
- Complete training and evaluation scripts"
```

### 7. Push to GitHub
```bash
# Push to main branch
git branch -M main
git push -u origin main
```

---

## 📝 Update README with Google Drive Links

After uploading to Google Drive:

1. **Get share links** for each folder
2. **Update README.md** with actual links:

```markdown
### Download Links

#### Datasets
- CASIA-Iris-Thousand: [Download (2.5 GB)](https://drive.google.com/file/d/ACTUAL_FILE_ID)
- LFW Funneled: [Download (200 MB)](https://drive.google.com/file/d/ACTUAL_FILE_ID)

#### Pre-trained Models (Included in Repository)
✅ Models already included - no separate download needed:
- `face_cnn/face_lfw_funneled_best.pth` (55 MB)
- `iris_cnn/iris_cnn_resnet18.pth` (46 MB)
- Embeddings files also included
```

3. **Commit the updated README**:
```bash
git add README.md
git commit -m "Add Google Drive download links"
git push
```

---

## 🎯 Files on GitHub (What Gets Committed)

✅ **Code** (~50 files, ~5 MB total)
```
face_cnn/*.py                 # Training/evaluation scripts
iris_cnn/*.py                 # Training/evaluation scripts
realtime_demo/*.py            # Demo applications
*.py                          # Root scripts
```

✅ **Configuration**
```
.gitignore
requirements.txt
README.md
IMPROVEMENT_REPORT.md
DEPLOYMENT_GUIDE.md
```

✅ **Documentation**
```
README.md
DEPLOYMENT_GUIDE.md
requirements.txt
.gitignore
```

❌ **Excluded** (in .gitignore)
```
CASIA-Iris-Thousand/          # Dataset → Too large for GitHub
lfw_funneled/                 # Dataset → Too large for GitHub
lfw-funneled.tgz              # Dataset archive
lfw_face/                     # Old dataset folder
venv/                         # Virtual environment
__pycache__/                  # Python cache
*.pyc                         # Compiled Python
```

⚠️ **Note**: Models (*.pth, *.npz) ARE included in repository for easy setup

---

## 🔄 Maintenance Workflow

### Update Code
```bash
git add .
git commit -m "Update: description of changes"
git push
```

### Update Models (after retraining)
1. Upload new *.pth to Google Drive
2. Update version in README.md
3. Commit README changes

---

## 📊 Repository Statistics

**Expected GitHub repo size**: ~150-200 MB (code + models)
**Google Drive storage**: ~2.7 GB (datasets only - optional)

**Clone time**: ~5-10 minutes (includes models)
**Full setup time**: ~10 minutes (no dataset download needed for demo)

---

## ✅ Deployment Checklist

- [x] Create .gitignore file
- [x] Create comprehensive README.md
- [x] Create requirements.txt
- [x] Train and save models
- [x] Clean up unnecessary files
- [ ] (Optional) Upload datasets to Google Drive for others
- [ ] Initialize Git repository
- [ ] Create GitHub repository  
- [ ] Push code + models to GitHub
- [ ] Test clone + setup on clean machine
- [ ] Add GitHub topics: `biometric`, `face-recognition`, `iris-recognition`, `pytorch`, `resnet18`, `multimodal-authentication`

---

## 🎓 Best Practices

1. **Never commit large files** (>100 MB) to GitHub
2. **Use Git LFS** for files 50-100 MB (optional)
3. **Compress datasets** before uploading to Drive
4. **Version your models** (e.g., v1.0, v2.0)
5. **Document breaking changes** in commit messages
6. **Keep README updated** with latest performance metrics

---

## 🆘 Common Issues

**Issue**: Git push fails with "file too large"
**Solution**: Check .gitignore, remove file from tracking:
```bash
git rm --cached <large-file>
git commit -m "Remove large file from tracking"
```

**Issue**: Google Drive link not accessible
**Solution**: Change sharing to "Anyone with the link"

**Issue**: Slow GitHub clone
**Solution**: Normal - only code, datasets download separately

---

Ready to deploy! 🚀
