# 📤 Deployment Guide - GitHub & Google Drive

## 🗂️ File Organization

### Files to Upload to Google Drive

#### **Large Dataset Files** (~2.7 GB)
```
CASIA-Iris-Thousand/          # 2.5 GB - Iris dataset
lfw_funneled/                 # 200 MB - Face dataset
```

#### **Trained Models** (~170 MB)
```
face_cnn/
├── face_cnn_resnet18.pth              # 55 MB
├── face_cnn_resnet18_metadata.pth     # 1 KB
└── face_embeddings_resnet18.npz       # 27 MB

iris_cnn/
├── iris_cnn_resnet18.pth              # 46 MB
├── iris_cnn_resnet18_metadata.pth     # 1 KB
└── iris_embeddings_resnet18.npz       # 39 MB
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
│   ├── face_cnn_resnet18.pth
│   ├── face_cnn_resnet18_metadata.pth
│   └── face_embeddings_resnet18.npz
│
├── iris_models/
│   ├── iris_cnn_resnet18.pth
│   ├── iris_cnn_resnet18_metadata.pth
│   └── iris_embeddings_resnet18.npz
│
└── evaluation_results/
    ├── final_fusion_resnet18_resnet18.png
    ├── face_comparison_resnet18_vs_insightface.png
    ├── iris_evaluation_resnet18.png
    └── fusion_evaluation_resnet18.png
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
3. Description: "Face + Iris Recognition with ResNet18 - 100% Accuracy"
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

- Face Recognition: ResNet18, AUC=1.0, EER=0.0%
- Iris Recognition: ResNet18, AUC=1.0, EER=0.03%
- Multimodal Fusion: AUC=1.0, EER=0.0%
- Real-time demo with MediaPipe iris segmentation
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

#### Pre-trained Models
- Face Models: [Download (83 MB)](https://drive.google.com/drive/folders/ACTUAL_FOLDER_ID)
- Iris Models: [Download (87 MB)](https://drive.google.com/drive/folders/ACTUAL_FOLDER_ID)
- All-in-One Package: [Download (170 MB)](https://drive.google.com/drive/folders/ACTUAL_FOLDER_ID)
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

✅ **Small Data** (optional)
```
pairs.txt                     # LFW evaluation pairs
pairsDevTest.txt
pairsDevTrain.txt
```

❌ **Excluded** (in .gitignore)
```
*.pth                         # Models → Google Drive
*.npz                         # Embeddings → Google Drive
CASIA-Iris-Thousand/          # Dataset → Google Drive
lfw_funneled/                 # Dataset → Google Drive
venv/                         # Virtual environment
__pycache__/                  # Python cache
```

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

**Expected GitHub repo size**: ~5-10 MB (code only)
**Google Drive storage**: ~2.9 GB (datasets + models)

**Clone time**: ~1 minute (code only)
**Full setup time**: ~30 minutes (including downloads)

---

## ✅ Deployment Checklist

- [ ] Create .gitignore file
- [ ] Create comprehensive README.md
- [ ] Create requirements.txt
- [ ] Upload datasets to Google Drive
- [ ] Upload models to Google Drive
- [ ] Upload evaluation results to Google Drive
- [ ] Set Google Drive permissions to "Anyone with link"
- [ ] Copy Google Drive share links
- [ ] Update README.md with actual download links
- [ ] Initialize Git repository
- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Test clone + setup on clean machine
- [ ] Add GitHub topics: `biometric`, `face-recognition`, `iris-recognition`, `pytorch`, `resnet18`

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
