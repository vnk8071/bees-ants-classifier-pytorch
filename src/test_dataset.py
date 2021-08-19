import os
from PIL import Image
from torch.utils.data import Dataset


class BeeAndAntDatasetTest(Dataset):
    def __init__(self, base_dir, transform):
        self.dir = base_dir
        self.transform = transform

    def __len__(self):
        return (len(os.listdir(self.dir)))

    def __getitem__(self, index):
        files = os.listdir(self.dir)
        file = files[index]
        image_path = os.path.join(self.dir, file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, file
