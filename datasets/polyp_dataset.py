import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATASET_ROOT = "/content/polypgen_dataset/polypgen_dataset"
IMAGE_SIZE = 256
RANDOM_SEED = 42
NUM_CLIENTS = 5

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )
    mask = (mask > 0).astype(np.float32)
    return mask

def build_index(dataset_root):
    samples = []

    sequences = sorted([
        d for d in os.listdir(dataset_root)
        if d.startswith("seq")
    ])

    for seq in sequences:
        img_dir = os.path.join(dataset_root, seq, "images")
        mask_dir = os.path.join(dataset_root, seq, "masks")

        for fname in sorted(os.listdir(img_dir)):
            samples.append({
                "seq": seq,
                "image": os.path.join(img_dir, fname),
                "mask": os.path.join(mask_dir, fname)
            })

    return samples

class PolypGenDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = preprocess_image(sample["image"])
        mask = preprocess_mask(sample["mask"])

        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

train_val, test_samples = train_test_split(
    samples,
    test_size=0.15,
    random_state=RANDOM_SEED
)

train_samples, val_samples = train_test_split(
    train_val,
    test_size=0.15,
    random_state=RANDOM_SEED
)

print(f"Train: {len(train_samples)}")
print(f"Val:   {len(val_samples)}")
print(f"Test:  {len(test_samples)}")

def split_into_clients(samples, num_clients=5):
    seq_groups = {}
    for s in samples:
        seq_groups.setdefault(s["seq"], []).append(s)

    sequences = sorted(seq_groups.keys())
    client_buckets = [[] for _ in range(num_clients)]

    for i, seq in enumerate(sequences):
        client_buckets[i % num_clients].extend(seq_groups[seq])

    return client_buckets
