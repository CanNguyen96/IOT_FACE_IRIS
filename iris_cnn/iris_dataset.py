import os
import cv2
import torch
from torch.utils.data import Dataset

class IrisDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {}
        label_id = 0

        for subject in sorted(os.listdir(root_dir)):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            self.label_map[subject] = label_id
            label_id += 1

            for eye in ["L", "R"]:
                eye_dir = os.path.join(subject_path, eye)
                if not os.path.exists(eye_dir):
                    continue

                for img in os.listdir(eye_dir):
                    if img.lower().endswith(".jpg"):
                        self.samples.append(
                            (os.path.join(eye_dir, img), self.label_map[subject])
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 64))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img, label
