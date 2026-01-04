"""
Simple MFA Real-time Demo
Face + Iris Multi-Factor Authentication
Using FUSION-BASED DECISION: Only fused score determines authentication
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys, os
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "iris_cnn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "face_cnn"))

from iris_model_improved import IrisCNN_ResNet
from face_model_improved import FaceRecognitionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Fusion parameters from evaluation (99.95% accuracy)
FUSION_THRESHOLD = 0.33  # Final decision threshold
FUSION_WEIGHT_FACE = 0.3
FUSION_WEIGHT_IRIS = 0.7

# Check if enrolled data exists
enrolled_path = os.path.join(os.path.dirname(__file__), "enrolled_user.npz")
if not os.path.exists(enrolled_path):
    print(f"❌ ERROR: {enrolled_path} not found!")
    print("Run prepare_enrollment.py first to enroll a user")
    sys.exit(1)

# Load enrolled embeddings
enrolled = np.load(enrolled_path)
face_ref = enrolled['face']
iris_ref = enrolled['iris']
print(f"✓ Loaded enrolled user embeddings (face: {face_ref.shape}, iris: {iris_ref.shape})")

# Load Face ResNet18 model
print("[INFO] Loading Face ResNet18 model...")
face_checkpoint = os.path.join(os.path.dirname(__file__), "..", "face_cnn", "face_lfw_funneled_best.pth")
if not os.path.exists(face_checkpoint):
    print(f"❌ ERROR: {face_checkpoint} not found!")
    sys.exit(1)

checkpoint = torch.load(face_checkpoint, map_location=DEVICE)
num_classes_face = checkpoint['num_classes']
embedding_size = checkpoint['embedding_size']

face_model = FaceRecognitionModel(
    num_classes=num_classes_face,
    embedding_size=embedding_size,
    pretrained=False
)
face_model.load_state_dict(checkpoint['model_state_dict'])
face_model.to(DEVICE).eval()
print(f"[OK] Face model loaded (epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.2f}%)")

# Face preprocessing (same as training)
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Iris ResNet18 model
print("[INFO] Loading Iris ResNet18 model...")
iris_checkpoint = os.path.join(os.path.dirname(__file__), "..", "iris_cnn", "iris_cnn_resnet18.pth")
if not os.path.exists(iris_checkpoint):
    print(f"❌ ERROR: {iris_checkpoint} not found!")
    sys.exit(1)

# Iris model was trained with 1000 classes (from CASIA-Iris-Thousand)
num_classes_iris = 1000

iris_model = IrisCNN_ResNet(
    num_classes=num_classes_iris,
    embedding_dim=512,
    pretrained=False
)
iris_model.load_state_dict(torch.load(iris_checkpoint, map_location=DEVICE))
iris_model.to(DEVICE).eval()
print(f"[OK] Iris model loaded (1000 classes, 512 embedding)")

# Iris preprocessing (grayscale, normalized)
def preprocess_iris(img_crop):
    """Convert iris crop to model input tensor"""
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
    resized = cv2.resize(gray, (512, 64))
    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

print("\n" + "="*60)
print("MFA REAL-TIME DEMO - FUSION-BASED DECISION")
print("="*60)
print("Controls:")
print("  F - Capture FACE for verification")
print("  I - Capture IRIS for verification")
print("  ESC - Exit")
print(f"Fusion: {FUSION_WEIGHT_FACE}*Face + {FUSION_WEIGHT_IRIS}*Iris > {FUSION_THRESHOLD}")
print("="*60 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot open camera")
    sys.exit(1)

print("🎥 Camera ON - Press F for face, I for iris verification")

# Track if scores are captured (not verified individually)
face_captured = False
iris_captured = False
face_score = None
iris_score = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(display, "Face detected", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Display status
    y_offset = 30
    cv2.putText(display, "FUSION-BASED AUTHENTICATION", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 40
    
    # Face status
    face_color = (0, 255, 0) if face_captured else (128, 128, 128)
    face_text = f"Face: {'CAPTURED' if face_captured else 'NOT CAPTURED'}"
    if face_score is not None:
        face_text += f" (score: {face_score:.3f})"
    cv2.putText(display, face_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
    y_offset += 30
    
    # Iris status
    iris_color = (0, 255, 0) if iris_captured else (128, 128, 128)
    iris_text = f"Iris: {'CAPTURED' if iris_captured else 'NOT CAPTURED'}"
    if iris_score is not None:
        iris_text += f" (score: {iris_score:.3f})"
    cv2.putText(display, iris_text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, iris_color, 2)
    y_offset += 30
    
    # Fusion result - ONLY check when both scores are available
    if face_score is not None and iris_score is not None:
        fused_score = FUSION_WEIGHT_FACE * face_score + FUSION_WEIGHT_IRIS * iris_score
        y_offset += 10
        cv2.putText(display, f"Fused Score: {fused_score:.4f} (threshold: {FUSION_THRESHOLD})", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 40
        
        if fused_score > FUSION_THRESHOLD:
            cv2.putText(display, "ACCESS GRANTED", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(display, (5, 5), (display.shape[1]-5, display.shape[0]-5), (0, 255, 0), 5)
        else:
            cv2.putText(display, "ACCESS DENIED", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(display, (5, 5), (display.shape[1]-5, display.shape[0]-5), (0, 0, 255), 5)

    cv2.imshow("MFA Demo - Press F (face) or I (iris)", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ESC to exit
    if key == 27:
        break
    
    # F key - verify face
    elif key == ord('f') or key == ord('F'):
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            
            try:
                face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    face_emb = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
                
                # Normalize
                face_emb = face_emb / np.linalg.norm(face_emb)
                
                # Compare with enrolled
                face_score = cosine_similarity(
                    face_emb.reshape(1, -1),
                    face_ref.reshape(1, -1)
                )[0][0]
                
                face_captured = True
                print(f"Face captured! Score: {face_score:.4f}")
                
            except Exception as e:
                print(f"Face capture error: {e}")
                face_captured = False
                face_score = None
        else:
            print("No face detected!")
    
    # I key - verify iris (simplified - use eye region from face)
    elif key == ord('i') or key == ord('I'):
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Simple approximation: left eye is in upper-left quadrant of face
            eye_x = x + int(w * 0.2)
            eye_y = y + int(h * 0.25)
            eye_w = int(w * 0.3)
            eye_h = int(h * 0.2)
            
            # Make sure we don't go out of bounds
            eye_x = max(0, eye_x)
            eye_y = max(0, eye_y)
            eye_w = min(eye_w, frame.shape[1] - eye_x)
            eye_h = min(eye_h, frame.shape[0] - eye_y)
            
            iris_crop = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
            
            if iris_crop.size > 0:
                try:
                    iris_tensor = preprocess_iris(iris_crop)
                    
                    with torch.no_grad():
                        iris_emb = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
                    
                    # Normalize
                    iris_emb = iris_emb / np.linalg.norm(iris_emb)
                    
                    # Compare with enrolled
                    iris_score = cosine_similarity(
                        iris_emb.reshape(1, -1),
                        iris_ref.reshape(1, -1)
                    )[0][0]
                    
                    iris_captured = True
                    print(f"Iris captured! Score: {iris_score:.4f}")
                    
                except Exception as e:
                    print(f"Iris capture error: {e}")
                    iris_captured = False
                    iris_score = None
            else:
                print("Cannot extract iris region!")
        else:
            print("No face detected for iris extraction!")

cap.release()
cv2.destroyAllWindows()
print("\n👋 Demo ended")
