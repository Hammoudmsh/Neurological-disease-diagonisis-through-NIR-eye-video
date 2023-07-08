
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel('INFO')


import tensorflow, tensorflow_addons, segmentation_models 
#print(tensorflow.__version__, tensorflow_addons.__version__, segmentation_models.__version__)
import cv2
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import pandas as pd
import datetime
import pathlib
import dataframe_image as dfi
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import tensorflow_addons as tfa

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
#from keras.utils import generic_utils
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization,Dropout, Lambda
from keras import backend as K

from keras.callbacks import ModelCheckpoint    



import glob
import argparse
from pathlib import Path
import parsing_file
# import inference


#------------------------------user libraries
from utilis import utilitis, plot_img, ClearToBlack, read_color_map, load_data_all, prepare_data, encode_lables
# import imageProcessing
from ML_DL_utilis import MLDL_utilitis
import models as mnet
from models import jacard_coef, dice_coef, jacard_coef_loss, dice_coef_loss, jacard_coef_loss, preprocessing_model_data, get_compiled_model

mldl_uts = MLDL_utilitis()
uts = utilitis()



    




#SIZE = (160, 120, 3)

from utilis import load_json
SIZE = load_json("config.json")['input_size']
SIZE_X, SIZE_Y= SIZE[0], SIZE[1]
BATCH_SIZE_EVALUATE = load_json("config.json")['BS_eval']





batch_size_val, batch_size_test = 1, 1
testValRatio, testRatio = 0.30, 0.30
#BATCH_SIZE = 1
epoch_interval = 1
STEP_SIZE  =2

wanted = 'p'
DATASET = "ClinicAnnotated_DA"
imagedata, maskdata = load_data_all(DATASET, num_to_read = 0, SIZE = SIZE)
if DATASET == "NN_human_mouse_eyes":
  DATASET = '../Data/NN_human_mouse_eyes';
elif DATASET == "MOBIUS":
  DATASET = '../Data/Eye dataset/pairs';
elif DATASET == "s-openeds":
  DATASET = r"../Data/s-openeds";
elif DATASET == "ClinicAnnotated_DA":
  DATASET = r"../Data/ClinicAnnotated_DA"

s = np.random.randint(0, len(maskdata))

plot_img(maskdata[s])
plt.savefig("check_before_clearning.png")

print("------------------------------------------------------------------------read colormap:")
dsInfo, labels_color, unwanted, wanted_classes = read_color_map(file = DATASET + "/classes.json", maskdata = maskdata,  wanted = wanted)
n_classes = len(labels_color)
bg_color = np.array(dsInfo['maskColor']['Background'])


#print(wanted, unwanted, "__________________________________________")
for i, mask in enumerate(maskdata):
    for uw in unwanted:
        maskdata[i] = ClearToBlack(mask, list(uw)[1])



# s = np.random.randint(0, len(maskdata))
plot_img(maskdata[s])
plt.savefig(f"check_after_clearning_{wanted}.png")
maskdata = maskdata.copy()


X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(imagedata, maskdata, testValRatio, testRatio)

#y_train_cat, y_val_cat, y_test_cat, y_cat, d = encode_lables(y_train, y_val, y_test, maskdata, labels_color)

for i, data in enumerate(zip(X_test, y_test)):
  img, mask = data
  cv2.imwrite(f"testtt/images/{i}.png", img)
  cv2.imwrite(f"testtt/labels/{i}.png", mask)
  


