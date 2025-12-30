import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import sys, os
from sklearn.metrics.pairwise import cosine_similarity
from iris_segmentation import IrisSegmenter

# Path
sys.path.insert(0, os.path.join("..", "iris_cnn"))
sys.path.insert(0, os.path.join("..", "face_cnn"))
from iris_model_improved import get_iris_model
from face_utils import get_face_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimized thresholds from evaluation
FACE_THRESHOLD = 0.6
IRIS_THRESHOLD = 0.38  # From ResNet18 evaluation (EER threshold)
FUSION_THRESHOLD = 0.33  # From fusion evaluation

# Load enrolled embeddings
face_ref = np.load("enrolled/user001/face.npy")
iris_ref = np.load("enrolled/user001/iris.npy")

# Face ResNet18 model
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

# Iris model - ResNet18
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
print("🎥 Camera ON - Look at camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
            face_emb.reshape(1,-1),
            face_ref.reshape(1,-1)
        )[0][0]

        if face_score > FACE_THRESHOLD:
            # Real iris verification with MediaPipe
            iris_crop = iris_segmenter.segment_iris(frame, eye='left', target_size=(512, 64))
            
            if iris_crop is not None:
                iris_tensor = torch.tensor(iris_crop / 255.0, dtype=torch.float32)\
                                .unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    iris_emb = iris_model(iris_tensor, return_embedding=True).cpu().numpy()[0]
                iris_emb = iris_emb / np.linalg.norm(iris_emb)
                
                iris_score = cosine_similarity(
                    iris_emb.reshape(1,-1),
                    iris_ref.reshape(1,-1)
                )[0][0]

                # Optimized fusion weights (0.3 face / 0.7 iris)
                fused = 0.3*face_score + 0.7*iris_score

                if fused > FUSION_THRESHOLD:
                    cv2.putText(frame, "ACCESS GRANTED", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    cv2.putText(frame, f"Score: {fused:.2f}", (50,90),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                else:
                    cv2.putText(frame, "IRIS FAILED", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame, "NO IRIS DETECTED", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),2)
        else:
            cv2.putText(frame, "FACE FAILED", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("MFA Demo", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
