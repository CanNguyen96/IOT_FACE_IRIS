"""
Quick demo: Face ResNet18 + Iris ResNet18 authentication (IMPROVED)
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
from face_utils import get_face_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== CẬP NHẬT THRESHOLDS =====
FACE_THRESHOLD = 0.85      
IRIS_THRESHOLD = 0.75      
FUSION_THRESHOLD = 0.70    

# ===== QUALITY CHECK FUNCTIONS =====
def check_face_quality(face_crop):
    """Check if face crop is good quality"""
    h, w = face_crop.shape[:2]
    
    # Check size
    if h < 80 or w < 80:
        return False, "Face too small"
    
    # Check brightness
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 30 or brightness > 220:
        return False, f"Bad lighting ({brightness:.1f})"
    
    # Check blur
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        return False, f"Too blurry ({blur_score:.1f})"
    
    return True, "OK"

def check_iris_quality(iris_crop):
    """Check if iris crop is good quality"""
    if iris_crop is None:
        return False, "No iris detected"
    
    # Check contrast
    std = np.std(iris_crop)
    if std < 20:
        return False, f"Low contrast ({std:.1f})"
    
    # Check if too bright/dark
    mean = np.mean(iris_crop)
    if mean < 30 or mean > 220:
        return False, f"Bad exposure ({mean:.1f})"
    
    return True, "OK"

print("="*70)
print("MULTIMODAL BIOMETRIC DEMO - IMPROVED SECURITY")
print("="*70)
print(f"Device: {DEVICE}")
print(f"\nSecurity Thresholds:")
print(f"  Face:   {FACE_THRESHOLD}")
print(f"  Iris:   {IRIS_THRESHOLD}")
print(f"  Fusion: {FUSION_THRESHOLD}")

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

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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

print("\n" + "="*70)
print("ENROLLMENT PHASE")
print("="*70)

cap = cv2.VideoCapture(0)
print("Press SPACE to enroll user (quality checks enabled)")

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
        # Capture face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            
            # Quality check
            is_good, msg = check_face_quality(face_crop)
            if not is_good:
                print(f"[WARN] Face quality check failed: {msg}")
                cv2.putText(vis_frame, f"Bad face: {msg}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Enrollment", vis_frame)
                cv2.waitKey(1500)
                continue
            
            face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                enrolled_face = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
            enrolled_face = enrolled_face / np.linalg.norm(enrolled_face)
            print("[OK] Face enrolled (quality: good)")
        else:
            print("[WARN] No face detected")
            continue
        
        # Capture iris
        iris_crop = iris_segmenter.segment_iris(frame, eye='left', target_size=(512, 64))
        
        # Quality check
        is_good, msg = check_iris_quality(iris_crop)
        if not is_good:
            print(f"[WARN] Iris quality check failed: {msg}")
            cv2.putText(vis_frame, f"Bad iris: {msg}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Enrollment", vis_frame)
            cv2.waitKey(1500)
            continue
        
        if iris_crop is not None:
            iris_tensor = torch.tensor(iris_crop / 255.0, dtype=torch.float32)\
                            .unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                enrolled_iris = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
            enrolled_iris = enrolled_iris / np.linalg.norm(enrolled_iris)
            print("[OK] Iris enrolled (quality: good)")
        
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
print("AUTHENTICATION PHASE (Enhanced Security)")
print("="*70)
print("Press SPACE to authenticate, ESC to exit")

verification_history = []
REQUIRED_CONSECUTIVE = 1  # Yêu cầu 1 lần pass

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display info
    info_text = f"Required: {REQUIRED_CONSECUTIVE} consecutive passes"
    cv2.putText(frame, "SPACE: Authenticate | ESC: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, info_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        # Test face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_score = 0
        face_quality_ok = False
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            
            # Quality check
            is_good, msg = check_face_quality(face_crop)
            if is_good:
                face_quality_ok = True
                face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    face_emb = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
                face_emb = face_emb / np.linalg.norm(face_emb)
                face_score = cosine_similarity(
                    face_emb.reshape(1, -1),
                    enrolled_face.reshape(1, -1)
                )[0][0]
            else:
                print(f"[WARN] Face quality: {msg}")
        
        # Test iris
        iris_crop = iris_segmenter.segment_iris(frame, eye='left', target_size=(512, 64))
        
        iris_score = 0
        iris_quality_ok = False
        
        is_good, msg = check_iris_quality(iris_crop)
        if is_good:
            iris_quality_ok = True
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
            print(f"[WARN] Iris quality: {msg}")
        
        # Fusion
        fused_score = 0.3 * face_score + 0.7 * iris_score
        
        # Quality gate
        if not (face_quality_ok and iris_quality_ok):
            print("\n" + "-"*70)
            print("❌ QUALITY CHECK FAILED")
            print(f"Face quality: {'OK' if face_quality_ok else 'FAIL'}")
            print(f"Iris quality: {'OK' if iris_quality_ok else 'FAIL'}")
            verification_history.clear()
            cv2.putText(frame, "QUALITY CHECK FAILED", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            # Show detailed quality status
            face_status = "OK" if face_quality_ok else "FAIL"
            iris_status = "OK" if iris_quality_ok else "FAIL"
            face_color = (0, 255, 0) if face_quality_ok else (0, 0, 255)
            iris_color = (0, 255, 0) if iris_quality_ok else (0, 0, 255)
            cv2.putText(frame, f"Face quality: {face_status}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)
            cv2.putText(frame, f"Iris quality: {iris_status}", (50, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, iris_color, 2)
            cv2.imshow("Result", frame)
            cv2.waitKey(2000)
            continue
        
        # Results
        print("\n" + "-"*70)
        print(f"Face Score:   {face_score:.4f} (threshold: {FACE_THRESHOLD})")
        print(f"Iris Score:   {iris_score:.4f} (threshold: {IRIS_THRESHOLD})")
        print(f"Fused Score:  {fused_score:.4f} (threshold: {FUSION_THRESHOLD})")
        
        # Record result
        passed = fused_score > FUSION_THRESHOLD
        verification_history.append(passed)
        
        # Keep only recent history
        if len(verification_history) > REQUIRED_CONSECUTIVE:
            verification_history = verification_history[-REQUIRED_CONSECUTIVE:]
        
        # Count passed
        passed_count = sum(verification_history)
        
        # Final decision
        if len(verification_history) >= REQUIRED_CONSECUTIVE and all(verification_history):
            print(f"✅ AUTHENTICATION SUCCESS ({REQUIRED_CONSECUTIVE}/{REQUIRED_CONSECUTIVE} passed)")
            cv2.putText(frame, "ACCESS GRANTED", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            verification_history.clear()  # Reset for next authentication
        else:
            print(f"❌ AUTHENTICATION FAILED ({passed_count}/{REQUIRED_CONSECUTIVE} passed)")
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