import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import tensorflow, tensorflow_addons, segmentation_models 
# print(tensorflow.__version__, tensorflow_addons.__version__, segmentation_models.__version__)
import cv2
# from tensorflow.keras.metrics import MeanIoU
# from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
import pandas as pd
import datetime
import pathlib
import dataframe_image as dfi
import json
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import tensorflow_addons as tfa

# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import class_weight



from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.utils import normalize
# from tensorflow.keras.utils import to_categorical
#from keras.utils import generic_utils
# import segmentation_models as sm
# from sklearn.model_selection import train_test_split
# from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization,Dropout, Lambda
# from keras import backend as K

# from keras.callbacks import ModelCheckpoint    
import glob




def showSample(fromDir, where):
    
    for file in os.listdir(fromDir):
        f = os.path.join(fromDir, file)
        if os.path.isfile(os.path.join(fromDir, file)):
            if 'all' in file:
                total = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            elif 'iris' in file:
                iris = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            elif 'pupil' in file:
                pupil = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            elif 'sclera' in file:
                sclera = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     print(img.shape, iris.shape)
#     fig, ax = plt.subplots(1,1 , figsize = (4,7))
#     res = np.concatenate([img, pupil, iris, sclera, total], axis = 0)
    
    nr, nc = 5, 1
    fig, ax = plt.subplots(nr,nc , figsize = (4,7))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    ax[0].imshow(img);
    ax[1].imshow(iris, cmap = "gray");
    ax[2].imshow(pupil, cmap = "gray");
    ax[3].imshow(sclera, cmap = "gray");
    ax[4].imshow(total, cmap = "gray");
    plt.subplots_adjust(wspace=0, hspace=0)

    
    for i in range(nr):
        ax[i].axis('off')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

    fig.savefig(where)
    fig.savefig(f'../Results/to_import_to_latex/ch3/{where.split("/")[-1]}')
#     plt.close()


def fromCommon(root = "../Data/s-openeds/", imagesFolder = "synthetic" ,  masksFolder = "mask-withskin", name = "s-openeds", SIZE = (128, 128)):
    global where
    tmp = os.listdir(root)
    folders = []
    for i in tmp:
        if os.path.isdir(root + i):
            folders.append(i)

    
    images = []
    masks = []
    for fol in folders:
        imagesFn = sorted(glob.glob(f"{root}{fol}/{imagesFolder}/*.tif"))
        masksFn = sorted(glob.glob(f"{root}{fol}/{masksFolder}/*.tif"))
        # print(len(imagesFn))
        s = np.random.randint(0, len(imagesFn), min(len(imagesFn), 2))
        # print("s",s)

        for i in s:
            f = imagesFn[i]
            img = cv2.imread(f)
            img =  cv2.resize(img, SIZE)
            images.append(img)
        for i in s:
            f = masksFn[i]
            img = cv2.imread(f)
            img =  cv2.resize(img, SIZE)
            masks.append(img)
    total1 = np.concatenate(images, axis = 1)
    total2 = np.concatenate(masks, axis = 1)
    total = np.concatenate([total1, total2], axis = 0)
    cv2.imwrite(f"{where}/{name}.png", total)



from pathlib import Path
if __name__ == "__main__":
    global SIZE, where
    where =  "../Results/samples"
    pathlib.Path(f'{where}').mkdir(parents=True, exist_ok=True)#metrics
    

    H = 1
    SIZE = (128,128)

    #-------------------------------------------------------------------------------------------------------1
    # showSample(fromDir = '../Data/Eye dataset/pairs/0/', where = f'../Data/Eye dataset/model1_samples1.png' )
    # showSample(fromDir = '../Data/Eye dataset/pairs/0/', where = f'../Data/Eye dataset/model1_samples2.png' )
    #-------------------------------------------------------------------------------------------------------1

    root = "../Data/NN_human_mouse_eyes"
    ImagesHumans = sorted(glob.glob(f"{root}/Images/*.jpg"))
    ImagesMouse = sorted(glob.glob(f"{root}/mouse/*.jpg"))
    Masks = sorted(glob.glob(f"{root}/Masks/*.png"))
    for j in range(2):
        s1 = np.random.randint(0, len(ImagesHumans), 3)
        s2 = np.random.randint(0, len(ImagesMouse), 1)

        imgsNames2plot = []
        masksNames2plot = []
        for i in s1:
            imgsNames2plot.append(ImagesHumans[i])
            mn = f"{root}/Masks/{Path(ImagesHumans[i]).name.split('.')[0] + '.png'}"
            masksNames2plot.append(mn)

        for i in s2:
            imgsNames2plot.append(ImagesMouse[i])
            mn = f"{root}/Masks/{Path(ImagesMouse[i]).name.split('.')[0] + '.png'}"
            masksNames2plot.append(mn)

        images = []
        masks = []
        for i, imgFileName in enumerate(imgsNames2plot):
            img = cv2.imread(imgFileName)
            img =  cv2.resize(img, SIZE)
            images.append(img)

        for i, maskFileName in enumerate(masksNames2plot):
            mask = cv2.imread(maskFileName)
            mask =  cv2.resize(mask, SIZE)
            masks.append(mask)
        
        total1 = np.concatenate(images, axis = H)
        total2 = np.concatenate(masks, axis = H)
        total = np.concatenate([total1, total2], axis = 1 - H)

        cv2.imwrite(f"{where}/NN_human_mouse_eyes_samples{j}.png", total)
        
        #--------------------------------------------
    root = "../Data/Eye dataset/pairs"

    folders = os.listdir(root)
    s = np.random.randint(0, len(folders), 2)
    total = []
    for j, fol in enumerate(s):
        allParts = sorted(glob.glob(f"{root}/{fol}/*.png"))
        for fileName in allParts:
            if 'all' in fileName:
                together = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            elif 'iris' in fileName:
                iris = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            elif 'pupil' in fileName:
                pupil = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            elif 'sclera' in fileName:
                sclera = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(fileName)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pupil = np.dstack((pupil, pupil, pupil))
        iris = np.dstack((iris, iris, iris))
        sclera = np.dstack((sclera, sclera, sclera))
        pupil =  cv2.resize(pupil, SIZE)
        iris =  cv2.resize(iris, SIZE)
        sclera =  cv2.resize(sclera, SIZE)
        together =  cv2.resize(together, SIZE)
        img =  cv2.resize(img, SIZE)

        # print(img.shape, together.shape, pupil.shape, iris.shape, sclera.shape)
        c = np.concatenate([img, pupil, iris, sclera, together], axis = H)
        cv2.imwrite(f"{where}/MOBIUS{j}.png", c)

        total.append(c)
    total = np.concatenate(total, axis = 1 - H)
    cv2.imwrite(f"{where}/MOBIUS.png", total)

    #-------------------------------------------------------------------------------------
    fromCommon(root = "../Data/s-openeds/", imagesFolder = "synthetic" ,  masksFolder = "mask-withskin", name = "s-openeds", SIZE = (128, 128))
    fromCommon(root = "../Data/s-nvgaze/", imagesFolder = "synthetic" ,  masksFolder = "mask-withskin", name = "s-nvgaze", SIZE = (128, 128))
    fromCommon(root = "../Data/s-natural/", imagesFolder = "synthetic" ,  masksFolder = "mask-withskin", name = "s-natural", SIZE = (128, 128))


    















    #-------------------------------------------------------------------------------------------------------1


    #-------------------------------------------------------------------------------------------------------1



    