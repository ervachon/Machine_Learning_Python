# -*- coding: utf-8 -*-
"""
Created on 2016/02/03

@author: eriva

os.chdir("D:\\_GIT_\\Machine_Learning_Python\\Canevas")

"""

# system
import os, sys, math, re

# dataframe and array
import pandas as pd
import numpy as np

# save objects
import pickle as pkl

# graph
import matplotlib.pyplot as plt

# math random
from random import random

# regexp
import re

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, classification_report
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score
from sklearn.cross_validation import train_test_split, StratifiedKFold

# directory
data_raw_dir   = "data.raw"
images_dir     = "images"
pickle_dir     = "pickles"
submission_dir = "submission"
data_dir       = "data"

# csv file
data_raw          = "cmc.csv"
submission_data   = "submission.csv"
validation_data_X = "validation_X.csv"
validation_data_Y = "validation_Y.csv"
train_test_data   = "train_test.csv"
kfolds_data       = "kfolds.csv"

# pickle
submission_pkl   = "submission.pkl"
validation_pkl_X = "validation_X.pkl"
validation_pkl_Y = "validation_Y.pkl"
train_test_pkl   = "train_test.pkl"
kfolds_pkl       = "kfolds.pkl"

# paramcsv file
separator = ","
na_value = ""

# seed
np.random.seed(1994)

# param 
validation_ratio = 0.1 
kfold = 5