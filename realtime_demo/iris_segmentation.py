"""
Iris Segmentation Module
Sử dụng MediaPipe Face Mesh để detect iris region chính xác
"""
import cv2
import numpy as np
import mediapipe as mp

class IrisSegmenter:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Quan trọng: Enable iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Iris landmarks indices (MediaPipe)
        # Left eye iris: 468-473 (5 points)
        # Right eye iris: 473-478 (5 points)
        self.LEFT_IRIS = [469, 470, 471, 472]
        self.RIGHT_IRIS = [474, 475, 476, 477]
        
        # Eye region landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def segment_iris(self, frame, eye='both', target_size=(512, 64)):
        """
        Detect và crop iris region từ frame
        
        Args:
            frame: BGR image từ webcam
            eye: 'left', 'right', hoặc 'both'
            target_size: (width, height) cho resize
            
        Returns:
            - Nếu eye='both': dict {'left': iris_crop, 'right': iris_crop}
            - Nếu eye='left'/'right': numpy array hoặc None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract iris regions
        output = {}
        
        if eye in ['left', 'both']:
            left_crop = self._crop_eye_region(gray, landmarks, self.LEFT_EYE, w, h, target_size)
            if eye == 'left':
                return left_crop
            output['left'] = left_crop
        
        if eye in ['right', 'both']:
            right_crop = self._crop_eye_region(gray, landmarks, self.RIGHT_EYE, w, h, target_size)
            if eye == 'right':
                return right_crop
            output['right'] = right_crop
        
        return output
    
    def _crop_eye_region(self, gray, landmarks, eye_indices, img_w, img_h, target_size):
        """Crop và resize eye region"""
        # Lấy bounding box của eye
        points = []
        for idx in eye_indices:
            lm = landmarks.landmark[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            points.append([x, y])
        
        points = np.array(points)
        x, y, w, h = cv2.boundingRect(points)
        
        # Expand bounding box một chút
        margin = int(w * 0.3)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_w - x, w + 2*margin)
        h = min(img_h - y, h + 2*margin)
        
        # Crop
        eye_crop = gray[y:y+h, x:x+w]
        
        if eye_crop.size == 0:
            return None
        
        # Resize to target size
        eye_crop = cv2.resize(eye_crop, target_size)
        
        return eye_crop
    
    def visualize_iris(self, frame):
        """
        Vẽ iris landmarks lên frame để debug
        
        Returns:
            frame với iris landmarks được vẽ
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return frame
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Vẽ left iris (xanh lá)
        for idx in self.LEFT_IRIS:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Vẽ right iris (xanh dương)
        for idx in self.RIGHT_IRIS:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        
        return frame
    
    def __del__(self):
        self.face_mesh.close()


# ========== HELPER FUNCTION ==========
def extract_iris_for_cnn(frame, segmenter=None):
    """
    Wrapper function để extract iris cho IrisCNN
    
    Returns:
        iris_crop (512x64) hoặc None
    """
    if segmenter is None:
        segmenter = IrisSegmenter()
    
    result = segmenter.segment_iris(frame, eye='left', target_size=(512, 64))
    return result


if __name__ == "__main__":
    # Test script
    segmenter = IrisSegmenter()
    cap = cv2.VideoCapture(0)
    
    print("Press 'v' to visualize iris landmarks")
    print("Press 's' to save cropped iris")
    print("Press ESC to exit")
    
    visualize = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if visualize:
            frame = segmenter.visualize_iris(frame)
        
        cv2.imshow("Iris Segmentation Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('v'):
            visualize = not visualize
            print(f"Visualize: {visualize}")
        elif key == ord('s'):
            result = segmenter.segment_iris(frame)
            if result:
                cv2.imwrite("iris_left.jpg", result['left'])
                cv2.imwrite("iris_right.jpg", result['right'])
                print("✔ Saved iris crops")
    
    cap.release()
    cv2.destroyAllWindows()
