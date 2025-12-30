"""
Quick demo: Face ResNet18 + Iris ResNet18 authentication
Press SPACE to authenticate, ESC to quit
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys, os
from sklearn.metrics.pairwise import cosine_similarity

# Path
sys.path.insert(0, os.path.join("..", "iris_cnn"))
sys.path.insert(0, os.path.join("..", "face_cnn"))
from iris_model_improved import get_iris_model
from iris_segmentation import IrisSegmenter
from face_utils import get_face_model  # Face ResNet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("MULTIMODAL BIOMETRIC DEMO - ResNet18 + ResNet18")
print("="*70)
print(f"Device: {DEVICE}")

# Load models
print("\n[INFO] Loading Face ResNet18 model...")
face_model = get_face_model(
    backbone='resnet18',
    num_classes=5749,
    embedding_dim=512,
    pretrained=False
)
face_model.load_state_dict(torch.load("../face_cnn/face_cnn_resnet18.pth", map_location=DEVICE))
face_model.to(DEVICE).eval()
print("[OK] Face ResNet18 loaded")

# Face preprocessing transform
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("[OK] Face detector loaded")

print("\n[INFO] Loading Iris ResNet18 model...")
iris_model = get_iris_model(
    backbone='resnet18',
    num_classes=1000,
    embedding_dim=512,
    pretrained=False
)
iris_model.load_state_dict(torch.load("../iris_cnn/iris_cnn_resnet18.pth", map_location=DEVICE))
iris_model.to(DEVICE).eval()
print("[OK] Iris ResNet18 loaded")

iris_segmenter = IrisSegmenter()

# Thresholds from evaluation
FACE_THRESHOLD = 0.6
IRIS_THRESHOLD = 0.38
FUSION_THRESHOLD = 0.33

print("\n" + "="*70)
print("ENROLLMENT PHASE")
print("="*70)

cap = cv2.VideoCapture(0)
print("Press SPACE to enroll user")

enrolled_face = None
enrolled_iris = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    vis_frame = iris_segmenter.visualize_iris(frame.copy())
    cv2.putText(vis_frame, "SPACE: Enroll | ESC: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Enrollment", vis_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        # Capture face with Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                enrolled_face = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
            enrolled_face = enrolled_face / np.linalg.norm(enrolled_face)
            print("[OK] Face enrolled")
        
        # Capture iris
        iris_crop = iris_segmenter.segment_iris(frame, eye='left', target_size=(512, 64))
        if iris_crop is not None:
            iris_tensor = torch.tensor(iris_crop / 255.0, dtype=torch.float32)\
                            .unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                enrolled_iris = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
            enrolled_iris = enrolled_iris / np.linalg.norm(enrolled_iris)
            print("[OK] Iris enrolled")
        
        if enrolled_face is not None and enrolled_iris is not None:
            print("\n[OK] Enrollment complete! Starting authentication...")
            break
    elif key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Authentication phase
print("\n" + "="*70)
print("AUTHENTICATION PHASE")
print("="*70)
print("Press SPACE to authenticate, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.putText(frame, "SPACE: Authenticate | ESC: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        # Test face with Face ResNet18
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                face_emb = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
            face_emb = face_emb / np.linalg.norm(face_emb)
            face_score = cosine_similarity(
                face_emb.reshape(1, -1),
                enrolled_face.reshape(1, -1)
            )[0][0]
        else:
            face_score = 0
        
        # Test iris
        iris_crop = iris_segmenter.segment_iris(frame, eye='left', target_size=(512, 64))
        if iris_crop is not None:
            iris_tensor = torch.tensor(iris_crop / 255.0, dtype=torch.float32)\
                            .unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                iris_emb = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
            iris_emb = iris_emb / np.linalg.norm(iris_emb)
            iris_score = cosine_similarity(
                iris_emb.reshape(1, -1),
                enrolled_iris.reshape(1, -1)
            )[0][0]
        else:
            iris_score = 0
        
        # Fusion (0.3 face + 0.7 iris - optimized)
        fused_score = 0.3 * face_score + 0.7 * iris_score
        
        # Results
        print("\n" + "-"*70)
        print(f"Face Score:   {face_score:.4f} (threshold: {FACE_THRESHOLD})")
        print(f"Iris Score:   {iris_score:.4f} (threshold: {IRIS_THRESHOLD})")
        print(f"Fused Score:  {fused_score:.4f} (threshold: {FUSION_THRESHOLD})")
        
        if fused_score > FUSION_THRESHOLD:
            print("✅ AUTHENTICATION SUCCESS")
            cv2.putText(frame, "ACCESS GRANTED", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            print("❌ AUTHENTICATION FAILED")
            cv2.putText(frame, "ACCESS DENIED", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.putText(frame, f"Fused: {fused_score:.3f}", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(2000)
    
    elif key == 27:  # ESC
        break
    
    cv2.imshow("Authentication", frame)

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Demo finished")
