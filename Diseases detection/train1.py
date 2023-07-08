
#!/usr/bin/env python
# coding: utf-8

# In[1]:

#from tensorflow.keras.metrics import MeanIoU
#from sklearn.preprocessing import MinMaxScaler,StandardScaler
#import seaborn as sns
#from skimage.measure import label, find_contours
#from skimage.color import label2rgb
#from skimage.measure import block_reduce

#from numpy import mean
#from numpy import std
#from numpy import dstack
#import datetime
#import dataframe_image as dfi
#import tensorflow_addons as tfa
#from tensorflow.keras.utils import normalize, to_categorical
#from keras.utils import generic_utils
#import segmentation_models as sm
#from keras.models import Model
#from keras import layers
#from keras import backend as K
#from torch.utils.data import Dataset, DataLoader
#import re
#import shutil
# import cv2
#import matplotlib
# from matplotlib.widgets import Slider
# import tensorflow.keras as keras
#import torch
#from torch import nn
#from torchvision import datasets
#from torch.utils.tensorboard import SummaryWriter
# from torchvision.transforms import ToTensor
# from tqdm.notebook import tqdm
#from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor
# from sklearn.metrics import confusion_matrix
# from tflite_model_maker import audio_classifier
# from torchvision import datasets
# from torch.utils.tensorboard import SummaryWriter
# import torchvision
# from PIL import Image, ImageChops
# from glob import glob
# from torch.utils.data import Dataset, DataLoader
# from torch import nn
# from torchvision import datasets
# import torchvision
# from keras.optimizers import adam_v2
#from tensorflow.keras.optimizers import Adam # - Works
# from keras.utils.np_utils import to_categorical
# import dill
# import weakref
#from torch.utils.tensorboard import SummaryWriter
#import torch.optim as optim
# from pytorchtools import EarlyStopping
#from torch.autograd import Variable
#from termcolor import colored


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
import cv2



from keras.layers import TimeDistributed
from keras.layers import ConvLSTM2D, LeakyReLU
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,\
                        concatenate, Conv2DTranspose, BatchNormalization,\
                        Dropout, Lambda
import tensorflow_addons as tfa
from keras.models import Sequential
from sklearn.utils import class_weight              

import numpy as np
import pandas as pd
import pathlib
import sys 

import glob
import time
import os
import random
import pickle
from tqdm.auto import tqdm
import json
from matplotlib import pyplot as plt
np.random.seed(2023)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#!/usr/bin/env python
# coding: utf-8
# # libraries
plt.rcParams["axes.labelsize"] = 'large'
plt.rcParams["axes.titlecolor"] = 'black'
plt.rcParams["axes.titlesize"] = 'large'
#plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 14
plt.rcParams['axes.titlepad'] = 10 
plt.style.use("ggplot")

# In[2]:

sys.path.insert(0, '../Pupil locator/')

#------------------------------user libraries
from utilis import utilitis
from ML_DL_utilis import MLDL_utilitis
from modelsArchs import get_specific_model
import parsing_file2
uts = utilitis()
mldl_uts = MLDL_utilitis()

# In[34]:


def readFile(fn):
    return pd.read_csv(fn) 

def getEvents(df, event_list = ['Saccade']):
    return df[df['event'].isin(event_list)]

def pad_split_chuncks(df, chunk_size = 4000, pad = False):
    def split_dataframe(df, chunk_size = 400): 
        chunks = list()
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks
    
    df_min_max_scaled = df.copy()
#     for column in df_min_max_scaled.columns:
#         df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    

    if pad:
        cur_len = len(df)
        num_pads = chunk_size - (cur_len%chunk_size)
        d = pd.DataFrame(np.zeros((num_pads, len(df.columns))), columns=df.columns)
#         df = df.append(pd.Series(0, index=df.columns), ignore_index=True)
        df = pd.concat([df_min_max_scaled, d], axis=0)
    parts = split_dataframe(df, chunk_size=chunk_size)
    return parts[0:-1]
def read_desired_data_images(df, desired_event, desired_features, read = -1, alg_kind = "RP", TS = 500, SIZE = 50):
    df = getEvents(df, event_list = desired_event)

    data_ = []
    labels_ = []
    events_ = []
    if read == -1:
        read = df.shape[0]
    for i in tqdm(range(read)):#
        #print(len(data_))
        record  = df.iloc[i]
        file = record["path2features"]
        disType = record["event"]
#         disType = record["event_enc"]
        target = record[["class_HC", "class_PD", "class_PSP"]].values        
        r = str(pathlib.Path(file).parent) + f"/all/{TS}/{SIZE}/{alg_kind}/" +str(pathlib.Path(file).stem) + "/"
        
        for feat in desired_features:
#             print(f"{r}{feat}*.png")
            x = glob.glob(f"{r}{feat}*.png")
#             files.extend(x)
#             print(x)
            for filename in x:
                tmp = cv2.imread(filename)
                
                if alg_kind in ["GASF", "GADF", "MTF","RP"] and len(tmp.shape) == 3:
                  tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)
                  tmp = np.expand_dims(tmp, axis = 2)
                tmp = tmp/255.0
                data_.append(tmp)
                labels_.append(target)
                events_.append(disType)            
    samples = list(zip(data_, labels_, events_))
    return np.array(samples)

"""
def read_desired_data_images(df, desired_event, desired_features, read = -1, alg_kind = "RP"):
    df = getEvents(df, event_list = desired_event)

    data_ = []
    labels_ = []
    events_ = []
    if read == -1:
        read = df.shape[0]
    for i in tqdm(range(read)):#
        #print(len(data_))
        record  = df.iloc[i]
        file = record["path2features"]
        disType = record["event"]
#         disType = record["event_enc"]
        target = record[["class_HC", "class_PD", "class_PSP"]].values        
        r = str(pathlib.Path(file).parent) + f"/{alg_kind}/" +str(pathlib.Path(file).stem) + "/"
        for feat in desired_features:
#             print(f"{r}{feat}*.png")
            x = glob.glob(f"{r}{feat}*.png")
#             files.extend(x)
#             print(x)
            for filename in x:
                tmp = cv2.imread(filename)
                tmp = tmp/255.0
                data_.append(tmp)
                labels_.append(target)
                events_.append(disType)            
    samples = list(zip(data_, labels_, events_))
    return np.array(samples)
"""




def read_desired_data(df, desired_event, desired_features, read = -1, SIZE = 7001, normalize = False):
    df = getEvents(df, event_list = desired_event)
    data_ = []
    labels_ = []
    events_ = []
    maxVal = 0
    """
    for i in tqdm(range(df.shape[0])):
        file = df.iloc[i]["path2features"]
        d = pd.read_csv(file, usecols = desired_features)
        maxVal = max(maxVal, len(d))
    print(maxVal)
    """
    maxVal = 3000
    if SIZE is not None:
      maxVal = SIZE
    
    for i in tqdm(range(df.shape[0])):
        if i == read:
            break
        record  = df.iloc[i]
        file = record["path2features"]
        disType = record["event"]
#         disType = record["event_enc"]
        
        target = record[["class_HC", "class_PD", "class_PSP"]].values
        d = pd.read_csv(file, usecols = desired_features)
        
        df_min_max_scaled = d.copy()
        if normalize:# done in 
            for column in df_min_max_scaled.columns:
                df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    


#         d = pad_split_chuncks(d, chunk_size = len(d), pad = True)[0]
#         tmp1 = []
#         for c in d.columns:
#             print(d[c].values)
#             cc = tf.keras.preprocessing.sequence.pad_sequences(np.array([d[c].values]), maxlen = len(d[c].values), padding='post', dtype ="float32")
#             print(cc)
#             tmp1.append(cc)
#         tmp = np.array(d.values)
        tmp1 = tf.keras.preprocessing.sequence.pad_sequences(np.array(df_min_max_scaled.T), maxlen = maxVal, padding='post', dtype ="float32", value = 2)
        data_.append(tmp1)
        labels_.append(target)
        events_.append(disType)
    samples = list(zip(data_, labels_, events_))
    return np.array(samples)#data_, labels_, events_

# print(df.columns)

# samples = read_desired_data(df.copy(), wanted, features, -1, SIZE)
# samples[0][0]


# In[35]:


# from pprint import pprint
# r = r'../Data/DataSet/dataFiles/GADF/HC_6_pursuit_M_52__years_part403/'
# # pprint()
# # pprint(glob.glob(r +'area*'+".png"))

# features = ["area", "c0_left"]
# files = []
# for feat in features:
#     print(f"{r}{feat}*.png")
#     x = glob.glob(f"{r}{feat}*.png")
# #     print(x)
#     files.extend(x)
# files


# In[36]:


# # fn = r"..\Data\DataSet\dataFilesBefore\HC_12_Gaze test_M_68__years_part146.csv"
# # df = readFile(fn)
# # df
# x = tf.keras.preprocessing.sequence.pad_sequences(np.array(df.T), maxlen = 2000, padding='post', dtype ="float32")
# x

def decode_features(s):
    features_name = {"X":"c0",
                 "Y":"c1",
                 "A":"area",
                 "J":"axis_major",
                 "N":"axis_minor"
                }
    #features = {"c0_left":0,"c1_left":0,"area":0,"axis_major_left":0,"axis_minor_left":0,"c0_right":0,"c1_right":0,"area.1":0, "axis_major_right":0, "axis_minor_right":0}
    features = []
    botheyes = s.split("_")
    if len(botheyes) == 2:
        left, right = botheyes
        for f in left:
            if features_name[f] != "area":
                #features[features_name[f]+"_left"] = 1
                features.append(features_name[f]+"_left")
            else:
                features.append(features_name[f])

        for f in right:
            if features_name[f] != "area":
                features.append(features_name[f]+"_right")
                #features[features_name[f]+"_right"] = 1
            else:
                features.append(features_name[f]+".1")
                #features[features_name[f]+".1"] = 1
    else:
        left = botheyes[0]
        for f in left:
            if features_name[f] != "area":
                #features[features_name[f]+"_left"] = 1
                features.append(features_name[f]+"_left")                
            else:
                features.append(features_name[f])
                #features[features_name[f]] = 1
    return features

def decode_tests(s):
    tests_name = {"N":"Spontaneous nystagmus",
                 "P":"pursuit",
                 "S":"Saccade",
                 "G":"Gaze test",
                 "O":"optokinetic"
                }
    tests = {"Spontaneous nystagmus":0,"pursuit":0,"Saccade":0,"Gaze test":0,"optokinetic":0}

    for t in s:
        tests[tests_name[t]] = 1
    return tests

# In[38]:
def readParametersFromCmd():
    global EPOCHS, LEARNING_RATE, EarlyStopping, wantedEvents, t, BATCH_SIZE, features, ts2img_algs, outputFile, SIZE, TS, USED_MODEL_ARCH, features_code, tests_code
    parser = parsing_file2.create_parser_disease_model()
    args = parser.parse_args()
    
    
    #BATCH_SIZE = 128
    #features = ["c0_left", "c1_left", "c0_right", "c1_right", "area", "area.1"]
    wantedEvents = {'Spontaneous nystagmus':1,
                'pursuit':0,
                'Saccade':0,
                'Gaze test':0,
                'optokinetic':0}
                #'position test':0,
    
    
    
    
    EPOCHS = args.epochs 
    EarlyStopping  = args.es
    LEARNING_RATE  = args.lr
    BATCH_SIZE = args.batch_size
    
    USED_MODEL_ARCH = args.USED_MODEL_ARCH
    
    t = args.file2read
    
    TS = args.TS
    SIZE = (50, 50, 3)
    SIZE = (args.SIZE, args.SIZE, 3) 
    TS = args.TS 
    #WANTED_ALGS = list(map(str, args.WANTED_ALGS.strip("[]").split(' ')))
    #WANTED_TESTS = list(map(str, args.WANTED_TESTS.strip("[]").split(' ')))
    #features = list(map(str, args.WANTED_FEATURES.strip("[]").split(',')))
    
    
    
    
    
    features = decode_features(args.WANTED_FEATURES)
    wantedEvents = decode_tests(args.WANTED_TESTS)
    features_code = args.WANTED_FEATURES
    tests_code  =args.WANTED_TESTS
    
    
    #features = []
    #for f, needed in args.WANTED_FEATURES.items():
    #  if needed:
    #     features.append(f)
         
    
    
     
    ts2img_algs = args.WANTED_ALGS
    
    
    #list(args.WANTED_ALGS.keys())[list(args.WANTED_ALGS.values()).index(1)]
    #ts2img_algs = []
    #for alg, needed in args.WANTED_ALGS.items():
    #  if needed:
    #     ts2img_algs.append(alg)

    
    #tmp = args.weights.strip("''").strip("[]")
    #class_weights = list(map(float, tmp.split(',')))
    outputFile = "" + args.output.strip("''")
    #DATASET = args.DS_NAME.strip("''")
    #model_weights = args.model_weights.strip("''")
    #metric_thr = args.metric_thr
    
    
    


readParametersFromCmd()


DETAILS_PATH = '../Data/DataSet/dataSetFile.csv'
batch_size_val, batch_size_test = 1, 1
testValRatio, testRatio = 0.30, 0.30
epoch_interval  =1
STEP_SIZE  =2

params = {'batch_size': BATCH_SIZE,
      'shuffle': True,
      'num_workers': 0}#0



#features = ["c0_left", "c1_left", "axis_major_left", "axis_minor_left", "area", "c0_right", "axis_major_right", "axis_minor_right", "area.1"]
#features = ["c0_left", "c1_left", "axis_major_left", "axis_minor_left", "area", "c0_right", "c1_right", "axis_major_right", "axis_minor_right", "area.1"]



current_model = outputFile#datetime.datetime.now().strftime("%d_%m_%Y(%H_%M_%S)")
current_model = f"../Results/model2/model_{current_model}"

print(current_model)
print("_________________________")
pathlib.Path(f'{current_model}/').mkdir(parents=True, exist_ok=True)#metrics
mldl_uts.setDir(d = f"{current_model}/")
# mldl_uts.setDir(d = f"{current_model}/metrics/")

cp_dir = f"{current_model}/checkpoints/"
cp_name = f"{current_model}/model2_results_best.hdf5"

model_results_plot_architecture = "model2_results_plot_architecture.png"
model_results_saved = "model2_results_saved.h5"
model_results_output = "model2_results_output.png"
model_results_architecture = "model2_results_architecture.png"


# In[39]:
print("---------------------------------------------------------------------Read All dataset")
df = pd.read_csv(DETAILS_PATH, index_col=False)
# uts.show(df, 5)
labelencoder = LabelEncoder()
df["label_enc"] = labelencoder.fit_transform(df["label"])
df["event_enc"] = labelencoder.fit_transform(df["event"])
# df.to_csv(f'{dest}\dataSetFileEnglish.csv', index = False)#, columns = ["x1","y1","d1","x2","y2","d2"])
df["path2features"] = df['path'].apply(lambda x:"../Data/DataSet/dataFiles/" + pathlib.Path(x).stem + ".csv")
df = pd.concat([df, pd.get_dummies(df.label, prefix='class')], axis=1)
# uts.show(df, 3)
df = df[df["valid"] == 1]
df = df[df["event"] != "position test"]
#uts.show(df, 3)



classes = np.unique(df["label"])#set(df["label"])
types  = np.unique(df["event"])#set`(df["event"])
noOfClasses = len(classes)
noOfDisType = len(types)
print(f"classes #{noOfClasses}: {classes}\ntypes #{noOfDisType}: {types}\n\t")

classes_dict = {}
for c in classes:
    classes_dict[df[df["label"] == c]["label_enc"].iloc[0]] = c

event_dict = {}
for e in types:
    event_dict[df[df["event"] == e]["event_enc"].iloc[0]] = e
classes_dict, event_dict
print("---------------------------------------------------------------------")
print("wantedEvents")
wanted = []
for k, v in wantedEvents.items():
    if v:
        print("\t",k)
        wanted.append(k)


#features = ["c0_left", "c1_left", "area", "axis_major_left", "axis_minor_left", "area", "c0_right", "c1_right", "axis_major_right", "axis_minor_right", "area.1"]

#samples = read_desired_data(df.copy(), wanted, features, -1, SIZE = 4000)
samples = read_desired_data_images(df, wanted, features, read = -1, alg_kind = ts2img_algs, TS = TS, SIZE = SIZE[0])

print(samples.shape)

print("---------------------------------------------------------------------Split dataset")
# # data = np.arange(df.shape[0])
# data = records

train_data, val_test_data  = train_test_split(samples,
                                    test_size = testValRatio,
                                    random_state = 42,
                                    shuffle = True)
val_data, test_data  = train_test_split(val_test_data,
                                    test_size = testRatio,
                                    random_state = 42,
                                    shuffle = True)



X_train = np.array(list(zip(*train_data))[0])
y_train = np.array(list(zip(*train_data))[1])

X_val = np.array(list(zip(*val_data))[0])
y_val = np.array(list(zip(*val_data))[1])

X_test = np.array(list(zip(*test_data))[0])
y_test = np.array(list(zip(*test_data))[1])




#X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
#X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])



# for i in train_data:
#     X_train.append()


# print("Train: ", (train_data.shape))

print("Train: ",X_train.shape, y_train.shape)
print("Val:   ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)




print(np.min(X_train[2][0]), np.max(X_train[2]))



#----------------------------------------------------------------------------------------------callback2
def scheduler1(epoch, lr):
    return lr if epoch < 800 else lr * tf.math.exp(-0.1)

def printLog(epoch, logs):
    print(logs)
    return
    if (epoch == 0):
        ss = f""
        for k in logs.keys():
            ss = ss + '{0}   '.format(k)
        print("Epochs\t",ss)
        
        
    if (epoch % 20) == 0:
        res = {key : np.round(logs[key], 3) for key in logs}
        s = f""    
        for  k in res.keys():
            s = s + '{0:04f}  '.format(res[k])
        print(epoch+1, '  ', s)

class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
            0 
                if epoch % self.epoch_interval != 0 
                else self.default_verbose
        )
        super().on_epoch_begin(epoch, *args, **kwargs)

callbacks =  [
      tf.keras.callbacks.LearningRateScheduler(scheduler1),
#     tfa.callbacks.TQDMProgressBar(leave_epoch_progress = True, leave_overall_progress = True, show_epoch_progress = False,show_overall_progress = True),
    tf.keras.callbacks.LambdaCallback(on_epoch_end = printLog),
    tf.keras.callbacks.ModelCheckpoint(cp_name, monitor='CategoricalAccuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True),
#     tf.keras.callbacks.ModelCheckpoint(cp_dir+cp_name, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="auto"),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)
    tf.keras.callbacks.ReduceLROnPlateau(monitor='CategoricalAccuracy', patience=20, verbose=0, factor=0.5, min_lr=0.00001),
#     tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
# tf.keras.callbacks.TensorBoard(log_dir=f"{current_model}/logs/fit/", histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1),
] 


my_metrics = [
     
#     "accuracy",
#     "sparse_categorical_accuracy",
     tf.keras.metrics.CategoricalAccuracy(name="CategoricalAccuracy"),
#     tf.keras.metrics.TruePositives(name='tp'),
#     tf.keras.metrics.FalsePositives(name='fp'),
#     tf.keras.metrics.TrueNegatives(name='tn'),
#     tf.keras.metrics.FalseNegatives(name='fn'), 
#     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
     tf.keras.metrics.Precision(name='precision'),
     tf.keras.metrics.Recall(name='recall'),
     #tf.keras.metrics.AUC(name='auc'),
     tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
     tf.keras.metrics.Precision(class_id=0, name="precision0"),
     tf.keras.metrics.Precision(class_id=1, name="precision1"),
     tf.keras.metrics.Precision(class_id=2, name="precision2"),
     tf.keras.metrics.Recall(class_id=0, name="recall0"),
     tf.keras.metrics.Recall(class_id=1, name="recall1"),
     tf.keras.metrics.Recall(class_id=2, name="recall2"),
     tfa.metrics.F1Score(num_classes=3, threshold=0.5)
]


# In[48]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = customFit(get_model())
# model = get_model(SIZE) 



print("-----------------------", SIZE, X_train[0].shape)

model = get_specific_model(name = USED_MODEL_ARCH, input_shape =  X_train[0].shape, num_classes = 3)


model_tmp1 = model
model.compile(#optimizer='adam'
              optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
#               loss=total,
              metrics=my_metrics,
             )#jit_compile = True)

model.summary()




def get_weights(y_cat):
    y = np.argmax(y_cat, axis = 1)

    # y_train = [1, 1,2,2,2,2,2,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
    classes = np.unique(y)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {k: v for k, v in zip(classes, weights)}
    print('Class weights:', class_weights)
    return class_weights

class_weight = get_weights(y_train.astype('float32'))
                

                                                 
start = time.time()

history = model.fit(x = X_train.astype('float32'),
                    y = y_train.astype('float32'),
                    epochs = EPOCHS,
#                     steps_per_epoch = steps_per_epoch_val,
                    validation_data = (X_val.astype('float32'), y_val.astype('float32')),
                    callbacks = callbacks,
                    shuffle = False,#True
                    verbose = 0,
                    class_weight = class_weight
                   )

end = time.time()
print("training time: ", end - start)


# # Results
print("---------------------------------------Results-----------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------")

_ = mldl_uts.saveModelArchitecture(model = model, fn = f"{model_results_architecture}", save = True);

results = model.evaluate(X_test.astype('float32'), y_test.astype('float32'), batch_size=32, return_dict= True, verbose=0)

mldl_uts.plotHistory(history, n = [1, 2], size = (10,5), show = False, prefix = "model2_results_")
tf.keras.utils.plot_model(model_tmp1,
                          to_file = f"{current_model}/{model_results_plot_architecture}",
                          show_shapes=True,
                          show_dtype=False,#do not
                          show_layer_names=False,#do not
                          rankdir='TB',
                          expand_nested=True,
                          dpi=96,
                          layer_range=None,
                         );


classification = model.predict(X_test.astype('float32'))
res = classification.argmax(axis=1)

resss = mldl_uts.make_confusion_matrix(
     y = np.argmax(y_test, axis = 1),
     y_pred = res,
 #     y = y_test,
 #     y_pred = predictions,
     group_names = ['True Neg','False Pos','False Neg','True Pos'],
     cmap = "jet",#"Greys",
     categories = classes,
     figsize = (9,7),
     title = "Confusion matrix",
     show = True, prefix = "model2_results_");



print(resss)


from utilis import append2csv
out = ''

#results = dict.fromkeys(order, 0)




#results["F1_0"] = 2 * resss[] * results["recall0"]/(results["precision0"] + results["recall0"])
#results["F1_1"] = 2 * results["precision1"] * results["recall1"]/(results["precision1"] + results["recall1"])
#results["F1_2"] = 2 * results["precision2"] * results["recall2"]/(results["precision2"] + results["recall2"])

F1_0,F1_1, F1_2 = results['f1_score']
del results['f1_score']

results['F1_0'] = F1_0
results['F1_1'] = F1_1
results['F1_2'] = F1_2


results["model_arch"] = USED_MODEL_ARCH
results["model_name"] = outputFile
results["Epochs"] = EPOCHS
results["lr"] = LEARNING_RATE
results["batch_size"] = BATCH_SIZE
results["SIZE"] = X_train[0].shape[0]
results["Tests"] = tests_code
results["features"] = features_code
results["imgAlg"] = ts2img_algs


order = results.keys()


#order = ["model_name","Epochs","lr", "SIZE", "Tests","imgAlg","trainAcc","trainLoss","trainPrecision","trainRecall","trainF1","valAcc","valLoss","valPrecision","valRecall","valF1","testAcc","testLoss","testPrecision","testRecall","testF1", "testAUC"]

"""
results["trainAcc"] = trainAcc
results["trainLoss"] = trainLoss
results["trainPrecision"] = trainPrecision
results["trainRecall"] = trainRecall
results["trainF1"] = EPOCHS

results["valAcc"] = EPOCHS
results["valLoss"] = EPOCHS
results["valPrecision"] = EPOCHS
results["valRecall"] = EPOCHS
results["valF1"] = EPOCHS


results["testAcc"] = Results["categorical_accuracy"]
results["testLoss"] = Results["categorical_accuracy"]
results["testPrecision"] = Results["precision"]
results["testRecall"] = Results["recall"]
results["testF1"] = Results["categorical_accuracy"]
results[""testAUC""] = Results["auc"]
"""

print(results)
df = pd.DataFrame.from_dict([results])
#df = df.drop([0,1], axis = 0)
print(df)
append2csv(f"{out}results_all_models.csv", df[order])

# In[ ]:
