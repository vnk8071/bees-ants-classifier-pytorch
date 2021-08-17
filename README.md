# bees-ants-classifier

Reference: https://github.com/python-engineer/pytorchTutorial/blob/master/15_transfer_learning.py

## Installation
Create virtual environment
```bash
conda create -n virtualenv python=3.7
conda activate virtualenv
```
Change into directory of project
```bash
cd bees-ants-classifier/
```
Install dependent packages
```bash
pip install -r requirements.txt
```

Download and set up data by
```bash
bash setup_data.sh
or
./setup_data.sh
```

Wait to download data

## Usage
Run
```bash
python train.py
```

Expected output:
```bash
-- Downloading ResNet18

Epoch 1/10
train - Loss: 0.8310 - Acc: 0.3689
val - Loss: 0.8525 - Acc: 0.3464
```
