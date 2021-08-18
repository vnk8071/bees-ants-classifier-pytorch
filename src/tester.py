import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from src.model import ResNet
from torchvision.transforms import transforms


class TestConfig(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key.upper(), args[key])


class Tester(TestConfig):
    def __init__(self, **args):
        super(Tester, self).__init__(**args)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        image_transformation = [
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ]
        self.image_transformation = transforms.Compose(image_transformation)

    def get_test_loader(self, test_image_dir):
        image_datasets = datasets.ImageFolder(os.path.join(
            test_image_dir), self.image_transformation)
        test_loader = DataLoader(
            dataset=image_datasets, batch_size=self.BATCH_SIZE, shuffle=None, num_workers=self.NUM_WORKERS)
        return test_loader

    def load_model(self):
        model = ResNet(self.NUM_CLASSES).to(self.device)
        model.load_state_dict(torch.load(self.CHECKPOINT_DIR)['model'])
        print('Load model success')
        return model

    def test(self):
        test_loader = self.get_test_loader(self.TEST_IMAGE_DIR)
        model = self.load_model()
        file_list = list()
        out_pred = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for index, (images, files) in enumerate(test_loader):
                images = images.to(self.device)

                pred = model(images)
                pred = pred.squeeze(1)
                out_pred = torch.cat((out_pred, pred), 0)
                file_list.append(files)
        return out_pred.to('cpu').tolist(), file_list

    def save_predict_csv(self):
        out_pred, file_list = self.test()
        # Save predict
        if not os.path.exists('./predicts'):
            os.makedirs('./predicts')
        df = pd.DataFrame({'image': file_list, 'predict': out_pred})
        df.to_csv(self.SAVE_PREDICTS_CSV, index=False)
        print('DONE')