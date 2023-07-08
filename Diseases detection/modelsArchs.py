
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')
# import cv2


#import seaborn as sns
#from skimage.measure import label, find_contours
#from skimage.color import label2rgb
#from skimage.measure import block_reduce

#from numpy import mean
#from numpy import std
#from numpy import dstack
# from keras.layers import TimeDistributed
# from keras.layers import ConvLSTM2D, LeakyReLU
from keras.layers import LSTM,Dense,Conv1D,MaxPooling1D,Flatten
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.preprocessing import LabelEncoder
# from matplotlib import pyplot as plt
import tensorflow as tf
# from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,\
                        concatenate, Conv2DTranspose, BatchNormalization,\
                        Dropout, Lambda
#from keras import Sequential
#from keras.utils import Sequence
from keras.layers import Masking
# from keras.layers import Bidirectional
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Flatten, Activation


from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
#from tensorflow.keras.metrics import MeanIoU
#from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
# import pandas as pd
# import pathlib
import sys 

# import glob
# import time
import os
# import random
# import pickle
# from tqdm.auto import tqdm
# import json
# from matplotlib import pyplot as plt
np.random.seed(2023)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# In[2]:

sys.path.insert(0, '../Pupil locator/')
#------------------------------user libraries

# In[2]:

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    num_classes = 3
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_specific_model(name, input_shape, num_classes = 3):
    if name == "TRANSFORMER":
        model = build_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
            num_classes = num_classes
        )
    elif name =="CNN":
        model = get_model2(input_shape, num_classes = 3)
    elif name =="LSTM":
        model = get_model3(input_shape, num_classes = 3)
    model.summary()
    return model
# In[2]:
def get_model1(input_shape, num_classes = 3):
    model = tf.keras.models.Sequential([
            #  First Convolution
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            # Second Convolution
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            # Third Convolution
            Conv2D(64, kernel_size=4, activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dropout(0.4),
            # Output layer
            Dense(3, activation='softmax')]
        )
    return model

def get_model2(input_shape, num_classes = 3):
    model = tf.keras.models.Sequential([
            #  First Convolution
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            # Second Convolution
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            # Third Convolution
            Conv2D(64, kernel_size=4, activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dropout(0.4),
            # Output layer
            Dense(3, activation='softmax')]
        )
    
#     input_x1 = Input(shape=input_shape)
#     input_y1 = Input(shape=input_shape)
#     input_a1 = Input(shape=input_shape)

#     input_x2 = Input(shape=input_shape)
#     input_y2 = Input(shape=input_shape)
#     input_a2 = Input(shape=input_shape)
    
#     input_x1_model = model(input_x1)
#     input_y1_model = model(input_y1)
#     input_a1_model = model(input_a1)

#     input_x2_model = model(input_x2)
#     input_y2_model = model(input_y2)
#     input_a2_model = model(input_a2)

#     conv = concatenate([input_x1_model, input_y1_model, input_a1_model, input_x2_model, input_y2_model, input_a2_model])
#     conv = Flatten()(conv)

#     dense = Dense(512)(conv)
#     dense = LeakyReLU(alpha=0.1)(dense)
#     dense = Dropout(0.5)(dense)

#     output = Dense(num_classes, activation='softmax')(dense)
#     model = Model(inputs=[input_x1, input_y1, input_a1, input_x2, input_y2, input_a2], outputs=[output])
    return model



# In[46]:
# # LSTM
# https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes
def get_model3(input_shape, num_classes = 3):
#     model = Sequential()
    #hidden layers is Ni + No * (2/3) -> 187 + 5 *(2/3) = 128
#     model.add(LSTM(128, input_shape=(187, 1), dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
#     model.add(Flatten())
#     model.add(Dense(5, activation='softmax')) #output of 5 potential encodings
#     return model
    (max_seq_len, dimension) = input_shape
    model2 = Sequential()
    model2.add(Masking(mask_value=2, input_shape=(max_seq_len, dimension)))
    #model2.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
    #model2.add(Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
    #model2.add(Bidirectional(LSTM(64, activation = 'sigmoid')))# kernel_regularizer = l2(0.01), recurrent_regularizer=l2(0.02), return_sequences = True, activation = 'sigmoid')))
    model2.add(LSTM(512, activation = 'relu'))
    #model2.add(Dropout(0.2))
    model2.add(Dense(num_classes, activation='softmax'))
    return model2
