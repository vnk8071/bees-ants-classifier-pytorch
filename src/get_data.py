import os
import shutil
import requests
from zipfile import ZipFile

URL = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
TARGET_PATH = 'data/hymenoptera_data.zip'


def download_file_from_pytorch(url, target_path):
    response = requests.get(url, stream=True)
    save_response_content(response, target_path)


def save_response_content(response, target_path):
    CHUNK_SIZE = 512

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    print('Start download data')
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    os.makedirs('./data')
    download_file_from_pytorch(URL, TARGET_PATH)
    print('Download done')

    with ZipFile('./data/hymenoptera_data.zip', 'r') as zip_file:
        zip_file.extractall('./data')
    print('Success unzip file')
