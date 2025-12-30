import os
import cv2
import torch
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {}
        label_id = 0

        for person in sorted(os.listdir(root_dir)):
            person_dir = os.path.join(root_dir, person)
            if not os.path.isdir(person_dir):
                continue

            self.label_map[person] = label_id
            label_id += 1

            for img in os.listdir(person_dir):
                if img.lower().endswith(".jpg"):
                    self.samples.append(
                        (os.path.join(person_dir, img), self.label_map[person])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return img, label
