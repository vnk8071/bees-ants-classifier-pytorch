# bees-ants-classifier

Organizing project: https://github.com/qnn122/organizing-training-project-tutorial

Reference: https://github.com/python-engineer/pytorchTutorial/blob/master/15_transfer_learning.py

Run project with CPU

Result of 10 epoch:
- Training complete in 3m 12s
- Best val Acc: 0.843137

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

Start trainig
Epoch 1/10
[TRAIN] - Loss: 0.8310 - Acc: 0.3689
[VAL] - Loss: 0.8525 - Acc: 0.3464
...
Done
```

## Prediction
Change the directory of TEST_IMAGES_DIR in test_config.json

After that, run
```bash
python test.py
```

Output csv file save into predicts folder

Let's try.
