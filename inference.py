"""
Capstone Project Team 5

Inference Pipeline

"""

import pandas as pd
import numpy as np
import pickle
import json

import cv2
import os

import random

# For transfer learning
import torch
import torch.nn as nn
# pretrained resnet
from torchvision.models import resnet50, ResNet50_Weights

# Classification Model
from sklearn.svm import SVC


def predict(img):
    
    # Initialize RESNET50
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Get all hidden layers before softmax, this will be the pretrained encoder generating the embeddings
    modules = list(resnet.children())[:-1]
    encoder = nn.Sequential(*modules)

    for param in encoder.parameters():
        # Freeze parameters so gradient is not computed in backward()
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
        param.requires_grad = False
    
    # Convert image
    im_pil = Image.fromarray(img)
    
    # Set to evaluation mode
    encoder.eval()

    # Initialize the Weight Transforms
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    # Apply it to the input image
    img_transformed = preprocess(im_pil)
    
    # embedding
    emb = encoder(img_transformed.unsqueeze(0)).squeeze().numpy()
    
    
    # Load Classification Model (SVM)
    with open('data/models/svm.pkl', 'rb') as f:
        clf = pickle.load(f)

    predictions = clf.predict_proba(model_input)[0]