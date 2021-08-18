from src.trainer import BeeAntClassifier
import json
from pprint import pprint

CONFIG_PATH = './config/train_config.json'
params = json.load(open(CONFIG_PATH, 'r'))
pprint(params)
model = BeeAntClassifier(**params)


def main():
    model.train_model()


if __name__ == '__main__':
    print('START TRAINING')
    main()
