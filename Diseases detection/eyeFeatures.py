import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')
import cv2
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import pandas as pd
import datetime
import pathlib
import dataframe_image as dfi


from torch.utils.tensorboard import SummaryWriter
import tensorflow_addons as tfa
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
from keras.utils import generic_utils

import segmentation_models as sm
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,\
                        concatenate, Conv2DTranspose, BatchNormalization,\
                        Dropout, Lambda

from keras import backend as K
from sklearn.preprocessing import LabelEncoder
import glob
print("Tensorflow version: ", tf. __version__)
from torch.utils.data import Dataset, DataLoader



import re
import shutil
# import cv2
import matplotlib.pyplot as plt
import matplotlib
import time
# from matplotlib.widgets import Slider
import os
# import tensorflow.keras as keras

import random


import torch
from torch import nn
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor
# from sklearn.metrics import confusion_matrix
# from tflite_model_maker import audio_classifier
# from torchvision import datasets
# from torch.utils.tensorboard import SummaryWriter
# import torchvision
import seaborn as sn

# from PIL import Image, ImageChops

# from glob import glob
import datetime

# from torch.utils.data import Dataset, DataLoader
# from torch import nn
# from torchvision import datasets
# import torchvision


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
# from keras.models import Sequential
# from keras import layers
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.optimizers import adam_v2
#from tensorflow.keras.optimizers import Adam # - Works

from sklearn.preprocessing import LabelEncoder
# from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
import pickle
# import dill
# import weakref
import random

from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
# from pytorchtools import EarlyStopping
from torch.autograd import Variable

# plt.rcParams["axes.labelsize"] = 'medium'
# plt.rcParams["axes.titlecolor"] = 'black'
# plt.rcParams["axes.titlesize"] = 'large'
# #plt.rcParams["figure.figsize"] = (15, 10)
# plt.rcParams["font.size"] = 14
# plt.rcParams['axes.titlepad'] = 10 
from termcolor import colored
import argparse


#!/usr/bin/env python
# coding: utf-8
# # libraries


# %% import standard libraries
import warnings
warnings.filterwarnings('ignore')
import pathlib
import re
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.measure import label, find_contours
from skimage.color import label2rgb
from skimage.measure import block_reduce
import argparse

plt.rcParams["axes.labelsize"] = 'large'
plt.rcParams["axes.titlecolor"] = 'black'
plt.rcParams["axes.titlesize"] = 'large'
#plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 14
plt.rcParams['axes.titlepad'] = 10 
plt.style.use("ggplot")

# #%% import user libraries
# from utilis import utilitis
# from ML_DL_utilis import MLDL_utilitis
# uts = utilitis()
# mldl_uts = MLDL_utilitis()
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
#import missingno as msno
import saccademodel







class eyeFeatures:
    
    def __init__(self):#, df, framerate = 50):
#         self.init_data(df, framerate = 50)
        pass
    
    def init_data(self, df, framerate = 50):
        self.df = df
        self.df_orginal = df

        self.xl, self.yl, self.axis_maj_left, self.axis_min_left, self.rl = self.df["c0_left"], self.df["c1_left"], self.df["axis_major_left"], self.df["axis_minor_left"], self.df["area"]
        self.xr, self.yr, self.axis_maj_right, self.axis_min_right, self.rr = self.df["c0_right"], self.df["c1_right"], self.df["axis_major_right"], self.df["axis_minor_right"], self.df["area.1"]


        self.framerate = framerate
        self.FS = 1/self.framerate
        
        
    def calcSaccadsRate(self, rawdata, framerate = 50):
        results = saccademodel.fit(rawdata)
        saccadic_reaction_time = len(results['source_points']) / framerate
        saccade_duration = len(results["saccade_points"]) / framerate
        return saccade_duration
    
    def calcSaccadsRate_df(self):
        SR_left = self.calcSaccadsRate(np.array([self.xl, self.yl]).T, framerate = 50)
        SR_right = self.calcSaccadsRate(np.array([self.xr, self.yr]).T, framerate = 50)
        return SR_left, SR_right

    def plot2D(self, x, y, xlbl, ylbl, label = "", title = "", ax1 = None, saveName = None):
        if ax1 is None:
            fig, ax1 = plt.subplots(1, 1, figsize = (10,3))
#         ax1.set_title(title)
        # Plot the data!
        ax1.plot(x, y, label = label) 
        ax1.set_xlabel(xlbl)
        ax1.set_ylabel(ylbl)
        ax1.legend()
        ax1.grid()

        if saveName is not None:
            plt.savefig(f'saveName.png')
            plt.legend()
            plt.show() 
        return ax1
    
    
    def plotXl(self, ax = None):
        ax1 = self.plot2D(x = np.arange(len(self.xl)), y = self.xl, title = "X", xlbl  = "Time", ylbl = "Left Gaze position along the x-axis", label="gaze point", ax1 = ax)
        return ax1
        
    def plotXr(self, ax = None):
        ax1 = self.plot2D(x = np.arange(len(self.xr)),y = self.xr, title = "X", xlbl  = "Time", ylbl = "Right Gaze position along the x-axis", label="gaze point", ax1 = ax)
        return ax1
        
    def plotYl(self, ax = None):
        ax1 = self.plot2D(x = np.arange(len(self.yl)), y = self.yl, title = "X", xlbl  = "Time", ylbl = "Left Gaze position along the y-axis", label="gaze point", ax1 = ax)
        return ax1
        
    def plotYr(self, ax = None):
        ax1 = self.plot2D(x = np.arange(len(self.yr)), y = self.yr, title = "X", xlbl  = "Time", ylbl = "Right Gaze position along the y-axis", label="gaze point", ax1 = ax)
        return ax1
    
    
    def plotLeftDiameter(self, ax):
        ax1 = self.plot2D(x = np.arange(len(self.xl)),y = self.rl, title = "Left Pupil size", xlbl  = "Time", ylbl = "Left Pupil size", label="", ax1 = ax)
        return ax1
    
    
    def plotRightDiameter(self, ax = None):
        ax1 = self.plot2D(x = np.arange(len(self.xr)),y = self.rr, title = "Right Pupil size", xlbl  = "Time", ylbl = "Right Pupil size", label="", ax1 = ax)
        return ax1
    
    
    def plot_spectrogramLeft(self, ax = None):
        fig, ax1 = plt.subplots(1, 1, figsize = (10,7), sharex=True)
        ax1.set_title("Spectrogram-Left")
        plt.specgram(self.rl, Fs = self.FS)    
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
#         plt.show()
        return ax1
    
    def plot_spectrogramRight(self, ax = None):
        fig, ax1 = plt.subplots(1, 1, figsize = (10,7), sharex=True)
        ax1.set_title("Spectrogram-Right")
        plt.specgram(self.rr, Fs = self.FS)    
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
#         plt.show() 
        return ax1
    
    def getVel(self, x):
        time = np.arange(len(x))
        dif = np.diff(x)
        tdif = np.diff(time)
        vel = dif / tdif
        return vel

    def getFeatures(self, show = False):
 
        velXl = self.getVel(self.xl)
        velYl = self.getVel(self.yl)
        velXr = self.getVel(self.xr)
        velYr = self.getVel(self.yr)
        
        pupil_left = self.rl.values
        pupil_right = self.rr.values
        
        SR_left = self.calcSaccadsRate(np.array([self.xl, self.yl]).T, framerate = 50)
        SR_right = self.calcSaccadsRate(np.array([self.xr, self.yr]).T, framerate = 50)
        
        if show:
            fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize = (15,9), sharex=True)
            fig, (ax8,ax9,ax10) = plt.subplots(3, 1, figsize = (15,9), sharex=True)
            
            ax1 = self.plotXl(ax1) 
            ax1 = self.plot2D(x = np.arange(len(velXl)), y = velXl, title = "Horizontal", xlbl  = "Time", ylbl = "Horizontal", label="Calculated velocity", ax1 = ax1)
            ax1.set_title("Left Gaze")
            ax1.grid()

            ax2 = self.plotYl(ax2) 
            ax2 = self.plot2D(x = np.arange(len(velYl)), y = velYl, title = "Vertical", xlbl  = "Time", ylbl = "Vertical", label="Calculated velocity", ax1 = ax2)
            ax2.grid()

            ax3 = self.plotLeftDiameter(ax3)

            
            
            
            
            ax8 = self.plotXl(ax8) 
            ax8 = self.plot2D(x = np.arange(len(velXr)), y = velXr, title = "Horizontal", xlbl  = "Time", ylbl = "Horizontal", label="Calculated velocity", ax1 = ax8)
            ax8.set_title("Right Gaze")
            ax8.grid()
            

            ax9 = self.plotXl(ax9) 
            ax9 = self.plot2D(x = np.arange(len(velYr)), y = velYr, title = "Vertical", xlbl  = "Time", ylbl = "Vertical", label="Calculated velocity", ax1 = ax9)
            ax9.grid()
            ax10  = self.plotRightDiameter(ax10)
            plt.legend()
#             ax1 = EF.plot_spectrogramLeft()
#             ax1 = EF.plot_spectrogramRight()
        
        res = {
            "velXl":velXl,
            "velYl":velYl,
            "velXr":velXr,
            "velYr":velYr,
            "pupil_left":pupil_left,
            "pupil_right":pupil_right,
            "SR_left":SR_left,
            "SR_right":SR_right
        }
        return res

    def immute(self, method, cols = None, k = 0, returnBack = 0):
        df = self.df
        df_imputed = df.copy()
        if cols is None:
            cols = df.columns
        if method == "linear":
            for c in cols:
                df_imputed[c] = df[c].interpolate(method='linear')
        elif method == "bfill":
            for c in cols:
                df_imputed[c] = df[c].fillna(method='bfill')
        elif method == "ffill":
            for c in cols:
                df_imputed[c] = df[c].fillna(method='ffill')
        elif method == "most_frequent":
            mode_imputer = SimpleImputer(strategy='most_frequent')
            for c in cols:
                df_imputed[c] = mode_imputer.fit_transform(df[c].values.reshape(-1,1))

        elif method == "knn":
            # Define scaler to set values between 0 and 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_knn = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

            # Define KNN imputer and fill missing values
            knn_imputer = KNNImputer(n_neighbors=k, weights='uniform', metric='nan_euclidean')
            df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), columns=df_knn.columns)
        elif method == "MICE":
            #Multivariate Imputation by Chained Equation — MICE
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            from sklearn import linear_model

#             df_mice = df.filter(['Distance','MaxSpeed','AvgSpeed','AvgMovingSpeed'], axis=1).copy()
            # Define MICE Imputer and fill missing values
            mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending')

            df_imputed = pd.DataFrame(mice_imputer.fit_transform(df), columns=cols)
        self.df = df_imputed

        self.xl, self.yl, self.axis_maj_left, self.axis_min_left, self.rl = self.df["c0_left"], self.df["c1_left"], self.df["axis_major_left"], self.df["axis_minor_left"], self.df["area"]
        self.xr, self.yr, self.axis_maj_right, self.axis_min_right, self.rr = self.df["c0_right"], self.df["c1_right"], self.df["axis_major_right"], self.df["axis_minor_right"], self.df["area.1"]




        if returnBack:
            return df_imputed
        