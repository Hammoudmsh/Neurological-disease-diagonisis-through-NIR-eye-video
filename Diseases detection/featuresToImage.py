#!/usr/bin/env python
# coding: utf-8

# In[66]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation
import pandas as pd
import glob
from tqdm.auto import tqdm
import pathlib
import cv2
import numpy as np
import tensorflow as tf


# In[ ]:


# def GAF(X):
#     X = np.reshape(X,(1,-1))
#     # Transform the time series into Gramian Angular Fields
#     gasf = GramianAngularField(image_size=21, method='summation')
#     X_gasf = gasf.fit_transform(X)
#     gadf = GramianAngularField(image_size=21, method='difference')
#     X_gadf = gadf.fit_transform(X)

#https://pyts.readthedocs.io/en/stable/generated/pyts.image.GramianAngularField.html#pyts.image.GramianAngularField    



def approximate_ts(X, window_size):
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    X_paa = paa.transform(X)
    return X_paa



def timeSeriesToImage(ts, size_x = None, kind = "GADF", window_size = 0):
    if window_size != 0:
        ts = approximate_ts(ts.reshape(1, -1) , window_size)
        ts = ts.reshape(-1,1)
    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')
    mtf = MarkovTransitionField()
    rp = RecurrencePlot()

    rp = RecurrencePlot()

    if kind == "GADF":
        img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "GASF":
        img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "MTF":
        img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
#         img = transformer.transform(ts)
    elif kind == "RP":
        img = rp.fit_transform(pd.DataFrame(ts).T)[0]
#         img = transformer.transform(ts)
    elif kind == "RGB_GAF":
        gasf_img = gasf.transform(pd.DataFrame(ts).T)[0]
        gadf_img = gadf.transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gasf_img,gadf_img,np.zeros(gadf_img.shape)))
    elif kind == "GASF_MTF":
        gasf_img = gasf.transform(pd.DataFrame(ts).T)[0]
        mtf_img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
        
        img = np.dstack((gasf_img,mtf_img, np.zeros(gasf_img.shape)))
    elif kind == "GADF_MTF":
        gadf_img = gadf.transform(pd.DataFrame(ts).T)[0]
        mtf_img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gadf_img,mtf_img, np.zeros(gadf_img.shape)))
    return img

def pad_split_chuncks(df, chunk_size = 600, pad = False):
    def split_dataframe(df, chunk_size = 600): 
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

# def create_gaf(ts, size_x = None):
#     """
#     :param ts:
#     :return:
#     """
#     data = dict()
#     if size_x is None:
#         size_x = ts.shape[0]
        
#     gadf = GramianAngularField(method='difference', image_size=size_x)
# #     data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0] # ts.T)
# #     return data
#     return gadf.fit_transform(pd.DataFrame(ts).T)[0] # ts.T)

# def split_signal(sig, n_samples, length):
#     """Split a signal into non-overlapping chunks.

#     Parameters
#     ----------
#     sig : 1d array
#         Time series.
#     n_samples : int
#         The chunk size to split the signal into, in samples.

#     Returns
#     -------
#     chunks : 2d array
#         The signal, split into chunks, with shape [n_chunks, chunk_size].

#     Notes
#     -----
#     If the signal does not divide evenly into the number of chunks, this approach
#     will truncate the signal, returning the maximum number of chunks, and dropping
#     any leftover samples.
#     """

#     n_chunks = int(np.floor(len(sig) / float(n_samples)))
#     chunks = np.reshape(sig[:int(n_chunks * n_samples)], (n_chunks, int(n_samples)))
#     res = []
#     for i, v in enumerate(chunks):
#         res.append(tf.keras.preprocessing.sequence.pad_sequences([chunks[i]], maxlen = length, padding='post'))
#     return np.array(res) 


# In[242]:


def convert_ts_to_images(DATASET_FILE, kind, features, minVal, window_size, saveTo):
    files = glob.glob(DATASET_FILE + "*.csv")
    total = 0
    for f in tqdm(files):
        pathlib.Path(DATASET_FILE  + kind + "/"+ pathlib.Path(f).stem).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(f, index_col = False, usecols = features)
        parts = [pad_split_chuncks(df.copy(), chunk_size = minVal, pad = True)[0]]
        for i, p in enumerate(parts):
            for feat in p.columns:
                # img = create_gaf(np.array(p[feat].values))
                img = timeSeriesToImage(np.array(p[feat].values), size_x =  None, kind = kind, window_size = window_size)
                pathlib.Path(saveTo+kind+"/"+ pathlib.Path(f).stem).mkdir(parents=True, exist_ok=True)
                #print(saveTo+kind+"/"+ pathlib.Path(f).stem)
                cv2.imwrite(saveTo+kind+"/"+ pathlib.Path(f).stem +"/" + feat+ "_part"+ str(i) + ".png", img*255)
                total += 1
    print(total)



features = ["c0_left"]#, "c1_left", "axis_major_left", "axis_minor_left", "area", "c0_right", "c1_right", "axis_major_right", "axis_minor_right", "area.1"]
#features = ["c0_left", "c1_left", "area", "c0_right", "c1_right", "area.1"]


#wanted_results ={500:[40, 50, 60, 70, 80, 90, 100], 750:[40, 50, 60, 70, 80, 90, 100]}
wanted_results ={500:[100]}

for  k, v in wanted_results.items():
    for wsize in v:
        print(f"Generating: {k} * {wsize}")
        to = f'../Data/DataSet/dataFiles/all/{k}/{wsize}/'
        x = k//wsize
        convert_ts_to_images("../Data/DataSet/dataFiles/", "GASF", features, k, x, to)
        convert_ts_to_images("../Data/DataSet/dataFiles/", "GADF", features, k, x, to)
        convert_ts_to_images("../Data/DataSet/dataFiles/", "MTF", features, k, x, to)
        convert_ts_to_images("../Data/DataSet/dataFiles/", "RP", features, k, x, to)
        convert_ts_to_images("../Data/DataSet/dataFiles/", "RGB_GAF", features, k, x, to)
        convert_ts_to_images("../Data/DataSet/dataFiles/", "GASF_MTF", features, k, x, to)
        convert_ts_to_images("../Data/DataSet/dataFiles/", "GADF_MTF", features, k, x, to)  

"""  
convert_ts_to_images("../Data/DataSet/dataFiles/", "GASF", features, 500)
convert_ts_to_images("../Data/DataSet/dataFiles/", "GADF", features, 500)
convert_ts_to_images("../Data/DataSet/dataFiles/", "MTF", features, 500)
convert_ts_to_images("../Data/DataSet/dataFiles/", "RP", features, 500)
"""
