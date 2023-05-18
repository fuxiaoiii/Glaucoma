import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchvision.models import resnet18
import timm
from torch.utils.data import DataLoader
from skimage.io import imread
import sklearn
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
from early_stopping import EarlyStopping
import os
from airogs_dataset import Airogs
import wandb
import sys
import sklearn.metrics
import yaml

torch.multiprocessing.set_sharing_strategy('file_system')
num_workers = 0
batch_size = 8

# original
model_0 = timm.create_model('efficientnet_b0', num_classes=2)
model_0.load_state_dict(torch.load('/home/wangqy/gardnet/Checkpoints/airogs_1.pt')['state_dict'])

# polar
model_1 = timm.create_model('efficientnet_b0', num_classes=2)
model_1.load_state_dict(torch.load('/home/wangqy/gardnet/Checkpoints/airogs_2.pt')['state_dict'])

# cropped
model_2 = timm.create_model('efficientnet_b0', num_classes=2)
model_2.load_state_dict(torch.load('/home/wangqy/gardnet/Checkpoints/airogs_3.pt')['state_dict'])

models = [model_0, model_1, model_2]

transforms = [
    torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC)]),
    torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC)]),
    torchvision.transforms.Compose([torchvision.transforms.ToTensor(), transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC)]),
]

apply_clahe = [True, True, True]
path = ['/home/wangqy/gardnet',
        '/home/wangqy/gardnet',
        '/home/wangqy/gardnet'
        ]

images_dir_name = ['dataset/*',
                   'dataset/*',
                   'dataset/*'
                   ]
test_datasets = [Airogs(path=path[0], images_dir_name=images_dir_name[0], split="test", transforms=transforms[0],
                        apply_clahe=apply_clahe[0]),
                 Airogs(path=path[1], images_dir_name=images_dir_name[1], split="test", transforms=transforms[1],
                        apply_clahe=apply_clahe[1], polar_transforms=True),
                 Airogs(path=path[2], images_dir_name=images_dir_name[2], split="test", transforms=transforms[2],
                        apply_clahe=apply_clahe[2]),
                 ]

test_loader = [
    DataLoader(test_datasets[0], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    DataLoader(test_datasets[1], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    DataLoader(test_datasets[2], batch_size=batch_size, shuffle=False, num_workers=num_workers)
]
# %%
labels = {0: [], 1: [], 2: []}
predictions = {0: [], 1: [], 2: []}
probs = {0: [], 1: [], 2: []}

with torch.no_grad():
    for i in range(3):
        models[i].eval()
        models[i] = models[i].cuda()
        for (inp, target) in tqdm(test_loader[i]):
            labels[i] += target
            batch_prediction = models[i](inp.cuda())
            probs[i] += torch.softmax(batch_prediction, dim=1)
            _, batch_prediction = torch.max(batch_prediction, dim=1)
            predictions[i] += batch_prediction.detach().tolist()

_probs = {}
_labels = {}

_probs[0] = np.asarray(list(map(lambda item: item.cpu().numpy(), probs[0])))
_probs[1] = np.asarray(list(map(lambda item: item.cpu().numpy(), probs[1])))
_probs[2] = np.asarray(list(map(lambda item: item.cpu().numpy(), probs[2])))

_labels[0] = np.asarray(list(map(lambda item: item.cpu().numpy(), labels[0])))
_labels[1] = np.asarray(list(map(lambda item: item.cpu().numpy(), labels[1])))
_labels[2] = np.asarray(list(map(lambda item: item.cpu().numpy(), labels[2])))

#综合三个模型的表现
# %%
w_1 = 1
w_2 = 1.5
w_3 = 1
avg_probs = (w_1 * _probs[0] + w_2 * _probs[1] + w_3 * _probs[2]) / 3

#展示结果
preds = np.argmax(avg_probs, axis=1)
gt = _labels[0]
sklearn.metrics.f1_score(gt, preds, average="macro")
sklearn.metrics.roc_auc_score(gt, preds)
sklearn.metrics.confusion_matrix(gt,preds)

