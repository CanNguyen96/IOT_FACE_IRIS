"""
Script để test iris segmentation
Chạy để đảm bảo MediaPipe hoạt động đúng
"""
import sys
import os

# Kiểm tra dependencies
try:
    import cv2
    import numpy as np
    import mediapipe as mp
    print("[OK] OpenCV, NumPy, MediaPipe installed")
except ImportError as e:
    print(f"[FAIL] Missing dependency: {e}")
    print("\nInstall with:")
    print("pip install opencv-python mediapipe numpy")
    sys.exit(1)

# Import iris segmenter
try:
    from iris_segmentation import IrisSegmenter
    print("[OK] IrisSegmenter imported successfully")
except ImportError as e:
    print(f"[FAIL] Cannot import IrisSegmenter: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("IRIS SEGMENTATION TEST")
print("="*50)
print("\nControls:")
print("  'v' - Toggle visualization (show iris landmarks)")
print("  's' - Save cropped iris images")
print("  'c' - Show cropped iris in window")
print("  ESC - Exit")
print("\n" + "="*50 + "\n")

segmenter = IrisSegmenter()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[FAIL] Cannot open webcam!")
    sys.exit(1)

visualize = True
show_crop = False

print("[INFO] Webcam opened. Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[FAIL] Failed to read frame")
        break
    
    display = frame.copy()
    
    # Visualize iris landmarks
    if visualize:
        display = segmenter.visualize_iris(display)
    
    # Try to segment iris
    result = segmenter.segment_iris(frame, eye='both', target_size=(512, 64))
    
    # Status text
    status = "[OK] IRIS DETECTED" if result and result.get('left') is not None else "[FAIL] NO IRIS"
    color = (0, 255, 0) if result and result.get('left') is not None else (0, 0, 255)
    
    cv2.putText(display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(display, "Press 'v': Visualize | 's': Save | ESC: Exit", (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Iris Segmentation Test", display)
    
    # Show cropped iris if available
    if show_crop and result:
        if result.get('left') is not None:
            cv2.imshow("Left Iris Crop", result['left'])
        if result.get('right') is not None:
            cv2.imshow("Right Iris Crop", result['right'])
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        print("\n[OK] Test completed")
        break
    elif key == ord('v'):
        visualize = not visualize
        print(f"Visualization: {'ON' if visualize else 'OFF'}")
    elif key == ord('c'):
        show_crop = not show_crop
        print(f"Show crop: {'ON' if show_crop else 'OFF'}")
        if not show_crop:
            cv2.destroyWindow("Left Iris Crop")
            cv2.destroyWindow("Right Iris Crop")
    elif key == ord('s'):
        if result:
            if result.get('left') is not None:
                cv2.imwrite("test_iris_left.jpg", result['left'])
                print("[OK] Saved: test_iris_left.jpg")
            if result.get('right') is not None:
                cv2.imwrite("test_iris_right.jpg", result['right'])
                print("[OK] Saved: test_iris_right.jpg")
            if result.get('left') is None and result.get('right') is None:
                print("[FAIL] No iris detected to save")
        else:
            print("[FAIL] No iris detected to save")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*50)
print("TEST SUMMARY")
print("="*50)
print("If you saw green dots on your iris -> SUCCESS [OK]")
print("If no detection -> Try better lighting or closer to camera")
print("="*50)
