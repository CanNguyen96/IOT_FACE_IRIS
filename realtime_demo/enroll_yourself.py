"""
Enroll your own face and iris for MFA demo
Captures your embeddings from webcam
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys, os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "iris_cnn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "face_cnn"))

from iris_model_improved import IrisCNN_ResNet
from face_model_improved import FaceRecognitionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# Load Face ResNet18 model
print("[INFO] Loading Face ResNet18 model...")
face_checkpoint = os.path.join(os.path.dirname(__file__), "..", "face_cnn", "face_lfw_funneled_best.pth")
checkpoint = torch.load(face_checkpoint, map_location=DEVICE)

face_model = FaceRecognitionModel(
    num_classes=checkpoint['num_classes'],
    embedding_size=checkpoint['embedding_size'],
    pretrained=False
)
face_model.load_state_dict(checkpoint['model_state_dict'])
face_model.to(DEVICE).eval()
print(f"[OK] Face model loaded\n")

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
iris_model = IrisCNN_ResNet(num_classes=1000, embedding_dim=512, pretrained=False)
iris_model.load_state_dict(torch.load(iris_checkpoint, map_location=DEVICE))
iris_model.to(DEVICE).eval()
print(f"[OK] Iris model loaded\n")

# Iris preprocessing
def preprocess_iris(img_crop):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if len(img_crop.shape) == 3 else img_crop
    resized = cv2.resize(gray, (512, 64))
    normalized = resized / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

print("="*60)
print("ENROLLMENT - CAPTURE YOUR FACE AND IRIS")
print("="*60)
print("Step 1: Press SPACE to capture your FACE")
print("Step 2: Press SPACE to capture your IRIS (eye region)")
print("Press ESC to cancel")
print("="*60 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot open camera")
    sys.exit(1)

# Step 1: Capture face
face_embedding = None
print("👤 Step 1: Position your face in camera, press SPACE when ready...")

while face_embedding is None:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display, "Face detected - Press SPACE", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(display, "STEP 1: CAPTURE FACE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "Press SPACE when face is detected", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Enrollment", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("❌ Enrollment cancelled")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    elif key == 32:  # SPACE
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            
            try:
                face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    face_embedding = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
                # Normalize
                face_embedding = face_embedding / np.linalg.norm(face_embedding)
                print("✓ Face captured successfully!\n")
            except Exception as e:
                print(f"❌ Face capture failed: {e}")
        else:
            print("❌ No face detected! Try again...")

# Step 2: Capture iris
iris_embedding = None
print("👁 Step 2: Look at camera, press SPACE to capture iris...")

while iris_embedding is None:
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Eye region (approximation)
        eye_x = x + int(w * 0.2)
        eye_y = y + int(h * 0.25)
        eye_w = int(w * 0.3)
        eye_h = int(h * 0.2)
        
        eye_x = max(0, eye_x)
        eye_y = max(0, eye_y)
        eye_w = min(eye_w, frame.shape[1] - eye_x)
        eye_h = min(eye_h, frame.shape[0] - eye_y)
        
        cv2.rectangle(display, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), (0, 255, 255), 2)
        cv2.putText(display, "Eye region - Press SPACE", (eye_x, eye_y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    cv2.putText(display, "STEP 2: CAPTURE IRIS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "Look at camera, press SPACE", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Enrollment", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("❌ Enrollment cancelled")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    elif key == 32:  # SPACE
        if len(faces) > 0:
            iris_crop = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
            
            if iris_crop.size > 0:
                try:
                    iris_tensor = preprocess_iris(iris_crop)
                    with torch.no_grad():
                        iris_embedding = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
                    # Normalize
                    iris_embedding = iris_embedding / np.linalg.norm(iris_embedding)
                    print("✓ Iris captured successfully!\n")
                except Exception as e:
                    print(f"❌ Iris capture failed: {e}")
            else:
                print("❌ Cannot extract iris region! Try again...")
        else:
            print("❌ No face detected! Try again...")

cap.release()
cv2.destroyAllWindows()

# Save enrolled embeddings
output_path = os.path.join(os.path.dirname(__file__), "enrolled_user.npz")
np.savez(output_path, face=face_embedding, iris=iris_embedding)

print("="*60)
print("✓ ENROLLMENT COMPLETED!")
print("="*60)
print(f"Face embedding: {face_embedding.shape}")
print(f"Iris embedding: {iris_embedding.shape}")
print(f"Saved to: {output_path}")
print("\nNow you can run: py realtime_demo/simple_mfa_demo.py")
print("="*60)
