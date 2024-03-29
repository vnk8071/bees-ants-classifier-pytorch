﻿# Bees-Ants-classifier using Pytorch
This is a single classifier model to predict bee and ant image.

Some bee and ant images:

![Some bee and ant images](https://github.com/vnk8071/bees-ants-classifier-pytorch/blob/master/images/bees_ants.PNG)

<ins>Data collected from Pytorch</ins>: https://download.pytorch.org/tutorial/hymenoptera_data.zip

The directory of dataset structured:

```bash
hymenoptera_data/
├── train
│   ├── ants
│   │    ├── ant1_train.png
│   │    ├── ...
│   │    └── antn_train.png
│   └── bees
│        ├── bee1_train.png
│        ├── ...
│        └── been_train.png
└── val
    ├── ants
    │    ├── ant1_val.png   
    │    ├── ... 
    │    └── antn_val.png   
    └── bees
         ├── bee1_val.png     
         ├── ...       
         └── been_val.png
```

Method: Transfer learning by using Resnet 18 and just change parameter of last layer

Resnet 18 architecture

![Resnet18](https://github.com/vnk8071/bees-ants-classifier-pytorch/blob/master/images/resnet18.png)

Run project with CPU

Result of 10 epochs:
- Training complete in 3m 12s
- Best validate accuracy: 0.843137

## Table of content
* [Acknowledgements](#Acknowledgements)
* [Installation](#Installation)
* [Visualization](#Visualizaztion)
* [Usage](#Usage)
* [Prediction](#Prediction)

## Acknowledgements
<ins>Organizing project follow</ins>: https://github.com/qnn122/organizing-training-project-tutorial

<ins>Reference</ins>: https://github.com/python-engineer/pytorchTutorial/blob/master/15_transfer_learning.py

## Installation
Create virtual environment
```bash
conda create -n beeantcls python=3.7
conda activate beeantcls
```
Change into directory of project
```bash
cd bees-ants-classifier-pytorch/
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

Wait to download data.
```bash
Start download data:
Total size:  47286322
[################################] 92357/92357 - 00:00:13
Download done
Success unzip file
```

## Visualization
Visualize:
- Show images
- Graph of model 
- Result epoch of train and validate  

Use TensorBoard 
```bash
tensorboard --logdir=runs
```

After training:

Open TensorBoard API: http://localhost:6006/

## Usage
Run
```bash
python train.py
```

Expected output:
```bash
-- Downloading ResNet18

train_config file

Start trainig
Epoch 1/10
[TRAIN] - Loss: 0.8310 - Acc: 0.3689
[VAL] - Loss: 0.8525 - Acc: 0.3464
...
Done
```

## Prediction
Run
```bash
python test.py
```

Output csv file save into predicts folder.

Or you can use Streamlit API by:
```bash
streamlit run streamlit_app.py
```

With URL: http://localhost:8501/

And upload any image of bee and ant to predict.

Let's try.
