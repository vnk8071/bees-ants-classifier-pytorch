import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from PIL import Image


##### CONSTANTS #####
# Mean of ImageNet dataset (used for normalization)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
# Std of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]


#### Datasets ####
class BeeAntDataset:
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        self.image_paths = []
        self.image_labels = []
        image_transformation = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]
        self.image_transformation = transforms.Compose(image_transformation)
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(
            data_dir, x), self.image_transformation) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes

    def set_up_training_data(self, BATCH_SIZE, NUM_WORKERS):
        dataset_sizes = {x: len(self.image_datasets[x]) for x in [
            'train', 'val']}
        dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=BATCH_SIZE,
                                     shuffle=True, num_workers=NUM_WORKERS)
                       for x in ['train', 'val']}
        return dataset_sizes, dataloaders

    def __getitem__(self, index):
        for image, label in self.image_datasets:
            self.image_paths = self.image_paths.append(image)
            self.image_labels = self.image_labels.append(label)
        image_path = self.image_paths[index]
        image_data = Image.open(image_path).convert("RGB")
        image_data = self.image_transformation(image_data)
        return image_data, torch.FloatTensor(self.image_labels[index]), image_path

    def __len__(self):
        return len(self.image_paths)
