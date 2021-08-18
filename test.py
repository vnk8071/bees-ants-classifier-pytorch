import os
from src.tester import Tester
import json
from pprint import pprint

CONFIG_PATH = os.path.join('config', 'test_config.json')
params = json.load(open(CONFIG_PATH, 'r'))
pprint(params)


predicter = Tester(**params)


def main():
    predicter.save_predict_csv()


if __name__ == '__main__':
    main()
