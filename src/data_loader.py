import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset

class KneeDataset(Dataset):
    """
    Loads a dataset of knee X-ray images organized in folders by KL grade.
    """
    def __init__(self, data_root, grades_to_load, images_per_grade=None):
        self.data_root = data_root
        self.image_files = []
        self.labels = []
        print(f"Initializing dataset: Loading images for KL grades {grades_to_load} from {data_root}")

        for kl_grade in sorted(grades_to_load):
            kl_dir = os.path.join(data_root, str(kl_grade))
            if not os.path.exists(kl_dir):
                print(f"Warning: Directory for KL{kl_grade} not found: {kl_dir}. Skipping.")
                continue

            files = sorted(glob.glob(os.path.join(kl_dir, '*.png')))
            if not files:
                print(f"Warning: No PNG images found in {kl_dir}. Skipping grade {kl_grade}.")
                continue

            files_to_use = files[:images_per_grade] if images_per_grade is not None else files
            self.image_files.extend(files_to_use)
            self.labels.extend([kl_grade] * len(files_to_use))
            print(f" Found {len(files_to_use)} images for KL grade {kl_grade}.")

        if not self.image_files:
            raise RuntimeError(f"No images found for specified grades {grades_to_load} in {data_root}")
        print(f"Dataset initialized with {len(self.image_files)} images total.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (224, 224)), -1 # Return a placeholder

class SingleImageDataset(Dataset):
    """A dataset that loads only a single specified image."""
    def __init__(self, image_path, label):
        self.image_path = image_path
        self.label = label
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            Image.open(image_path).convert('RGB') # Test load
        except Exception as e:
            raise ValueError(f"Cannot load image {image_path}: {e}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("SingleImageDataset only has one item at index 0")
        image = Image.open(self.image_path).convert('RGB')
        return image, self.label

class TransformedDatasetWrapper(Dataset):
    """
    A wrapper to apply transformations and map labels for a given dataset subset.
    """
    def __init__(self, subset, transform, target_map={2: 0, 3: 1}):
        self.subset = subset
        self.transform = transform
        self.target_map = target_map

    def __getitem__(self, idx):
        image, original_label = self.subset[idx]
        if original_label == -1:
            return torch.zeros((3, 224, 224)), -1 # Use values from config

        transformed_image = self.transform(image)
        # Map the original KL grade (e.g., 2, 3) to a model-friendly index (e.g., 0, 1)
        label = self.target_map.get(original_label, -1)
        if label == -1:
            print(f"Warning: Original label {original_label} at index {idx} not in target_map {self.target_map}")
        return transformed_image, label

    def __len__(self):
        return len(self.subset)
