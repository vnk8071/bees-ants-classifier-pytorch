import os
import PIL.Image
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import ResNet
from torchvision.transforms import transforms
from src.test_dataset import BeeAndAntDatasetTest


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

    def get_test_loader(self):
        '''
        Input: Folder contain images with no labels
        Output: Split image follow format test_config with
            - Transform
            - Batch size: 4
            - Num workers: 0
        '''
        test_dataset = BeeAndAntDatasetTest(
            self.TEST_IMAGE_DIR, self.image_transformation)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE,
                                 shuffle=None, num_workers=self.NUM_WORKERS)
        return test_loader

    def load_model(self):
        model = ResNet(self.NUM_CLASSES).to(self.device)
        model.load_state_dict(torch.load(self.CHECKPOINT_DIR)['model'])
        print('Load model success')
        return model

    def test(self):
        test_loader = self.get_test_loader()
        model = self.load_model()
        file_list = list()
        predict_labels = list()
        # out_pred = torch.FloatTensor().to(self.device)
        with torch.no_grad():
            for index, (images, files) in enumerate(test_loader):
                images = images.to(self.device)

                pred = model(images)

                _, predicted = torch.max(pred, 1)
                classifier = self.CLASSES[predicted]
                predict_labels.append(classifier)
                file_list.append(files[0])
                '''
                out_pred = torch.cat((out_pred, pred), 0)
        return out_pred.to("cpu").tolist(), file_list
        '''
        return predict_labels, file_list

    def predict_image(self, image_input):
        '''
        Input: Image of bee and ant
        Output: Predcit label of that image
        '''
        model = self.load_model()
        with torch.no_grad():
            image = PIL.Image.open(image_input)
            image_tensor = self.image_transformation(image)
            image_tensor = image_tensor.unsqueeze(0)
            predict = model(image_tensor)
            _, predicted = torch.max(predict, 1)
            classifier = self.CLASSES[predicted]
        return classifier

    def save_predict_csv(self):
        '''
        Orther method
        labels = ['ant', 'bee']
        results = list()
        out_pred, file_list = self.test()
        for ps in out_pred:
            result = {'prob_ant': 0, 'prob_bee': 0, 'label': labels[0]}
            result['prob_ant'] = float(ps[0])
            result['prob_bee'] = float(ps[1])
            result['label'] = labels[0] if result['prob_ant'] > result['prob_bee'] else labels[1]
            results.append(result)
        '''
        predict_labels, file_list = self.test()

        # Save predict
        if not os.path.exists('./predicts'):
            os.makedirs('./predicts')
        df = pd.DataFrame({'image': file_list, 'predict': predict_labels})
        df.to_csv(self.SAVE_PREDICTS_CSV, index=False)
        print('DONE')
