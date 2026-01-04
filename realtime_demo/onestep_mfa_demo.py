"""
One-Step MFA Real-time Demo
Press SPACE to capture BOTH Face + Iris and authenticate immediately
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
FUSION_THRESHOLD = 0.33
FUSION_WEIGHT_FACE = 0.3
FUSION_WEIGHT_IRIS = 0.7

# Check if enrolled data exists
enrolled_path = os.path.join(os.path.dirname(__file__), "enrolled_user.npz")
if not os.path.exists(enrolled_path):
    print(f"❌ ERROR: {enrolled_path} not found!")
    print("Run enroll_yourself.py first to enroll a user")
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

# Face preprocessing
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

num_classes_iris = 1000

iris_model = IrisCNN_ResNet(
    num_classes=num_classes_iris,
    embedding_dim=512,
    pretrained=False
)
iris_model.load_state_dict(torch.load(iris_checkpoint, map_location=DEVICE))
iris_model.to(DEVICE).eval()
print(f"[OK] Iris model loaded (1000 classes, 512 embedding)")

# Iris preprocessing
def preprocess_iris(img_crop):
    """Convert iris crop to model input tensor"""
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
    resized = cv2.resize(gray, (512, 64))
    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

print("\n" + "="*70)
print("ONE-STEP MFA AUTHENTICATION")
print("="*70)
print("Press SPACE - Capture Face + Iris and authenticate immediately")
print("Press ESC - Exit")
print(f"Fusion: {FUSION_WEIGHT_FACE}*Face + {FUSION_WEIGHT_IRIS}*Iris > {FUSION_THRESHOLD}")
print("="*70 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot open camera")
    sys.exit(1)

print("🎥 Camera ON - Press SPACE to authenticate")

# Store last authentication result
last_result = None
last_face_score = None
last_iris_score = None
last_fused_score = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display, "Face detected - Press SPACE", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display header
    y_offset = 30
    cv2.putText(display, "ONE-STEP MFA AUTHENTICATION", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 40
    
    cv2.putText(display, "Press SPACE to authenticate", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    y_offset += 40
    
    # Display last result if available
    if last_result is not None:
        y_offset += 10
        cv2.putText(display, f"Face Score: {last_face_score:.4f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30
        
        cv2.putText(display, f"Iris Score: {last_iris_score:.4f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30
        
        cv2.putText(display, f"Fused Score: {last_fused_score:.4f} (threshold: {FUSION_THRESHOLD})", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 50
        
        if last_result == "GRANTED":
            cv2.putText(display, "ACCESS GRANTED", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.rectangle(display, (5, 5), (display.shape[1]-5, display.shape[0]-5), (0, 255, 0), 8)
        else:
            cv2.putText(display, "ACCESS DENIED", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.rectangle(display, (5, 5), (display.shape[1]-5, display.shape[0]-5), (0, 0, 255), 8)

    cv2.imshow("One-Step MFA Demo", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    # ESC to exit
    if key == 27:
        break
    
    # SPACE - authenticate with both face and iris
    elif key == 32:  # SPACE
        if len(faces) == 0:
            print("❌ No face detected! Please position your face in the camera.")
            last_result = None
            continue
        
        print("\n" + "="*70)
        print("🔍 AUTHENTICATING...")
        print("="*70)
        
        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        
        # Extract eye region for iris
        eye_x = x + int(w * 0.2)
        eye_y = y + int(h * 0.25)
        eye_w = int(w * 0.3)
        eye_h = int(h * 0.2)
        
        eye_x = max(0, eye_x)
        eye_y = max(0, eye_y)
        eye_w = min(eye_w, frame.shape[1] - eye_x)
        eye_h = min(eye_h, frame.shape[0] - eye_y)
        
        iris_crop = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
        
        try:
            # Process Face
            print("  [1/3] Processing face...")
            face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                face_emb = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
            face_emb = face_emb / np.linalg.norm(face_emb)
            face_score = cosine_similarity(
                face_emb.reshape(1, -1),
                face_ref.reshape(1, -1)
            )[0][0]
            print(f"        Face score: {face_score:.4f}")
            
            # Process Iris
            print("  [2/3] Processing iris...")
            iris_tensor = preprocess_iris(iris_crop)
            with torch.no_grad():
                iris_emb = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
            iris_emb = iris_emb / np.linalg.norm(iris_emb)
            iris_score = cosine_similarity(
                iris_emb.reshape(1, -1),
                iris_ref.reshape(1, -1)
            )[0][0]
            print(f"        Iris score: {iris_score:.4f}")
            
            # Fusion
            print("  [3/3] Computing fusion...")
            fused_score = FUSION_WEIGHT_FACE * face_score + FUSION_WEIGHT_IRIS * iris_score
            print(f"        Fused score: {fused_score:.4f} (threshold: {FUSION_THRESHOLD})")
            
            # Decision
            print("\n" + "="*70)
            if fused_score > FUSION_THRESHOLD:
                print("✅ RESULT: ACCESS GRANTED")
                last_result = "GRANTED"
            else:
                print("❌ RESULT: ACCESS DENIED")
                last_result = "DENIED"
            print("="*70 + "\n")
            
            # Store for display
            last_face_score = face_score
            last_iris_score = iris_score
            last_fused_score = fused_score
            
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            last_result = None

cap.release()
cv2.destroyAllWindows()
print("\n👋 Demo ended")
