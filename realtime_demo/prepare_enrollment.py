import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys, os
from iris_segmentation import IrisSegmenter

# Path
sys.path.insert(0, os.path.join("..", "iris_cnn"))
sys.path.insert(0, os.path.join("..", "face_cnn"))
from iris_model_improved import get_iris_model
from face_utils import get_face_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LOAD MODELS ==========
# Face ResNet18
print("[INFO] Loading Face ResNet18 model...")
face_model = get_face_model(
    backbone='resnet18',
    num_classes=5749,
    embedding_dim=512,
    pretrained=False
)
face_model.load_state_dict(torch.load("../face_cnn/face_cnn_resnet18.pth", map_location=DEVICE))
face_model.to(DEVICE).eval()
print("[OK] Face ResNet18 loaded")

# Face preprocessing
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iris CNN - ResNet18
print("[INFO] Loading Iris ResNet18 model...")
iris_model = get_iris_model(
    backbone='resnet18',
    num_classes=1000,
    embedding_dim=512,
    pretrained=False
)
iris_model.load_state_dict(torch.load("../iris_cnn/iris_cnn_resnet18.pth", map_location=DEVICE))
iris_model.to(DEVICE).eval()
print("[OK] Iris ResNet18 loaded")

# Iris segmenter
iris_segmenter = IrisSegmenter()

cap = cv2.VideoCapture(0)
print("Press SPACE to enroll")
print("(Make sure your eyes are clearly visible)")

while True:
    ret, frame = cap.read()
    
    # Visualize iris detection
    vis_frame = iris_segmenter.visualize_iris(frame.copy())
    cv2.putText(vis_frame, "SPACE: Enroll | ESC: Cancel", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Enroll", vis_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # SPACE
        break
    elif key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        print("❌ Enrollment cancelled")
        exit()

cap.release()
cv2.destroyAllWindows()

# ========== FACE EMBEDDING ==========
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
assert len(faces) > 0, "No face detected"

x, y, w, h = faces[0]
face_crop = frame[y:y+h, x:x+w]
face_tensor = face_transform(face_crop).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    face_emb = face_model(face_tensor, return_embedding=True).cpu().numpy()[0]
face_emb = face_emb / np.linalg.norm(face_emb)

# ========== IRIS (MEDIAPIPE DETECTION) ==========
iris_crop = iris_segmenter.segment_iris(frame, eye='left', target_size=(512, 64))

if iris_crop is None:
    print("❌ No iris detected! Please try again.")
    exit()

# Show detected iris
cv2.imshow("Detected Iris", iris_crop)
cv2.waitKey(1000)
cv2.destroyAllWindows()

iris_tensor = torch.tensor(iris_crop / 255.0, dtype=torch.float32)\
                .unsqueeze(0).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    iris_emb = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
iris_emb = iris_emb / np.linalg.norm(iris_emb)

# ========== SAVE ==========
np.savez("enrolled_user.npz", face=face_emb, iris=iris_emb)
print("✔ Enrollment saved")
