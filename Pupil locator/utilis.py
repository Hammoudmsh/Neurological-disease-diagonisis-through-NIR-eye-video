import csv
import pandas as pd
import dataframe_image as dfi
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from IPython.display import display
from tqdm import tqdm
import glob
import cv2
import os
import json


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow, tensorflow_addons, segmentation_models 
print(tensorflow.__version__, tensorflow_addons.__version__, segmentation_models.__version__)
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


#SIZE = (160, 120, 3)

#from utilis import load_json

import sys 
sys.path.insert(0, '../Pupil locator/')



class utilitis:
    def compareTwoList(self, a, b):
        matches = [idx for idx, item in enumerate(zip(a, b)) if item[0] == item[1]]
        matchesNum = len(matches)
        return matches, matchesNum

    def save2csv(self, fileName, data, cols, header = False):
        with open(fileName, 'w+', newline='', encoding='utf-8') as f:
            write = csv.writer(f, delimiter=',')
            if header:
                write.writerow(cols)
            write.writerows(data)
    def find_between(self, s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""

    def find_between_r(self, s, first, last ):
        try:
            start = s.rindex( first ) + len( first )
            end = s.rindex( last, start )
            return s[start:end]
        except ValueError:
            return ""

    def isContain(self, fileName, fileTypes):
            for ext in fileTypes:
                if  ext not in fileName:
                    return False
            return True

    def show(self, df, nr):
        with pd.option_context('display.max_rows', nr,
                           'display.max_columns', None,
                           'display.width', 800,
                           'display.precision', 3,
                           'display.colheader_justify', 'left'):
            display(df)
        
    def tensor(self, *x):
        tmp = []
        for i in x:
            tmp.append(np.array(i))
    #         tmp.append(tf.convert_to_tensor(i))
        return tmp

    def dataframeAsImage(self, d, path, rowNames, save, colsNames =None):
        df = pd.DataFrame(data=d, index = rowNames, columns = colsNames)
        if save:
            dfi.export(df, path)
        return df
    def showRow(self, display_list, title, size = None):
        # plt.figure()
        fig, ax = plt.subplots(1, len(display_list), figsize = size)
        for i in range(len(display_list)):    
            if display_list[i] is not None:
                ax[i].set_title(title[i])
                ax[i].imshow(tf.keras.utils.array_to_img(display_list[i]));
                ax[i].axis('off')
                plt.close()
        # plt.show()
        return fig
        df = pd.DataFrame(data=d, index = rowNames)
        if save:
            dfi.export(df, path)

    def display(self, display_list, idx = None, num = None, title =  None, size =(10, 10), show = True):    
        if len(display_list[0].shape) in [2,3] :
            f = self.showRow(display_list, title, size = size)
            return f
        else:
            if idx is  None and num is not None or num == 1:
                idx = np.random.randint(0, len(display_list[0]), num)
            fig, ax = plt.subplots(num, len(display_list), figsize = size)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            for j, i in enumerate(idx):
                if j ==0:
                    titles__ = title
                else:
                    titles__ = [""] * len(display_list)
                tmp = []
                for img in display_list:
                    if img is not None and i < len(img):
                        x = img[i]
                    else:
                        x = None
                    tmp.append(x)
                    
                for i in range(len(display_list)):    
                    if tmp[i] is not None:
                        ax[j][i].set_title(titles__[i])
                        if i  in [1,2]:
                            ax[j][i].imshow(tf.keras.utils.array_to_img(tmp[i]), cmap = 'jet');
                        else:
                            ax[j][i].imshow(tf.keras.utils.array_to_img(tmp[i]));
                        ax[j][i].axis('off')
                        ax[j][i].set_aspect('equal')

                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                        if show == False:
                            plt.close()
            return fig





def append2csv(filename, df):
    df.to_csv(filename, mode='a', index=False, header = not Path(filename).exists())
            

def to3D(img, SIZE = None):
    # if SIZE is not None:
    # img =  cv2.resize(img, (64,64))

    if img.ndim == 2:
        # img = np.expand_dims(img, axis=2)
        img = np.dstack((img, img, img))
    return img


def plot_img(img, title="", cmap= "gray", figsize = (5,5)):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.title(title)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def getROI(img_rgb, maskColor):
    return cv2.inRange(img_rgb, maskColor, maskColor)

def ClearToBlack(img_rgb, maskColor, backgrround_color = [0, 0, 0]):    
#     x = 255 - cv2.inRange(img_rgb, np.array(maskColor)-5, np.array(maskColor)+5)
#     img_rgb[:, :, 0] = cv2.bitwise_and(img_rgb[:, :, 0], x)
#     img_rgb[:, :, 1] = cv2.bitwise_and(img_rgb[:, :, 1], x)
#     img_rgb[:, :, 2] = cv2.bitwise_and(img_rgb[:, :, 2], x)
#     r = np.array(list(map(test, img_rgb== maskColor)))
#     r, c = np.where(r)
#     r = list(r[:])
#     c = list(c[:])
# #     list(zip(r, c))
#     img_rgb[r[0:], c[0:], :] = [0, 0, 0]
    # print(maskColor)
    img_rgb[:, :, maskColor.index(255)] = 0
#     img_rgb[:, :, maskColor.index(1)] = 0
    #img_rgb[np.all(img_rgb == maskColor, axis=-1)] = 0
    return img_rgb


def replace_(txt, my_dict):
    res = []
    for i in txt:
        res.append(my_dict[i])
    return res  
def readClassesData(fn = "classes.json"):
    dsInfo = load_json(fn)
    return dsInfo#, np.array(pupil_color), np.array(sclera_color), np.array(iris_color), np.array(bg_color)





def load_data1(root = '../Data/Eye dataset/pairs', num_to_read = 0, SIZE = (160, 120, 3), ext = ['png', 'png']):
    imagearray = []
    maskarray=[]
    idx = 0
    t = num_to_read
    if t == 0:
        t = len(list(pathlib.Path(root).rglob(f"*.{ext[0]}")))//5

    #bar = tqdm(desc="Progress", ncols=100, ascii = '*')
    for path,subs,files in tqdm(os.walk(root), desc="Reading dataset ",  total = t, ncols=100):

        if idx == t:
            break
        if path != root:
            vv = os.listdir(path)
            imgName = min(vv, key = len)
            maskName = imgName.split('.')[0]+f"_all.{ext[0]}"
            image=cv2.imread(path+"/"+imgName)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
            if SIZE[2] == 1 and image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
            image=cv2.resize(image,( SIZE[0], SIZE[1]))
            """ Image processing """
            #image =  image.astype(np.float32) / 255.0
            #image = image.astype(np.float32)
            image=np.array(image)
            imagearray.append(image)
            #-----------------------------------------------------------------
            mask =cv2.imread(path+"/"+maskName)
            if mask.ndim == 3:
                pass
                #mask =cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
            
            mask =cv2.resize(mask,( SIZE[0], SIZE[1]))
            mask =np.array(mask)
            maskarray.append(mask)
            idx += 1
        #bar.update()
    imagedata = np.array(imagearray)
    maskdata =  np.array(maskarray)
    # imagedata = np.expand_dims(normalize(imagedata, axis=1), axis=3)
    return imagedata, maskdata



def load_data(root = '../Data/NN_human_mouse_eyes', num_to_read = 0, SIZE = (160, 120, 3), ext = ['jpg', 'png']):
    # Images = glob.glob(f"{root}/Images/*")
    # ext1 = {'jpg':0,'jpeg':0, 'png':0, 'tiff':0, 'tif':0}
    # for i in Images:
    #     tmp = (Path(i).name.split(".")[1]).lower()
    #     ext1[tmp] += 1

    # print(ext1)
    # Images = sorted(glob.glob(f"{root}/Masks/*"))
    # ext2 = {'jpg':0,'jpeg':0, 'png':0, 'tiff':0, 'tif':0}
    # for i in Images:
    #     tmp = (Path(i).name.split(".")[1]).lower()
    #     ext2[tmp] += 1

    # x = list(ext1.items())
    # x1 = list(zip(*x))[1]
    # im = np.argmax(np.array(x1))
    # imgExt = x[im][0]

    # x = list(ext2.items())
    # x1 = list(zip(*x))[1]
    # im = np.argmax(np.array(x1))
    # maskExt = x[im][0]

    # print(imgExt, maskExt)
    # ext = [imgExt, maskExt]

    #---------------------------------------------------------

    imagearray = []
    maskarray=[]
    t = num_to_read

#     len(list(pathlib.Path(root).rglob("*.jpg")))
    Images = sorted(glob.glob(f"{root}/Images/*.{ext[0]}"))
    if t == 0:
        t = len(Images)#len(list(pathlib.Path(root).rglob("*.jpg")))
    Images = Images[0:t]
    #     Masks = sorted(glob.glob(f"{root}/Masks/*.png")[0:t])
    for i in tqdm(range(t), desc = "Reading dataset"):
        img_tmp = cv2.imread(Images[i])
        #img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)     

        p = Path(Images[i])
        maskName = f"{root}/Masks/" + p.name.split(".")[0]+"."+ext[1]
        #print(maskName)
        mask_tmp = cv2.imread(maskName)
        #print(mask_tmp)
#         mask_tmp = cv2.cvtColor(mask_tmp, cv2.COLOR_BGR2GRAY)
#         _, mask_tmp = cv2.threshold(mask_tmp, 0, 255, cv2.THRESH_BINARY)

        
        if SIZE[2] == 1 and img_tmp.ndim == 3:
            img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)    
        #else:
        
        #mask_tmp = cv2.cvtColor(mask_tmp, cv2.COLOR_BGR2RGB)
             

        img_tmp = cv2.resize(img_tmp, (SIZE[0], SIZE[1]))
        mask_tmp = cv2.resize(mask_tmp, (SIZE[0], SIZE[1]))
            
        imagearray.append(img_tmp)
        maskarray.append(mask_tmp)
        
    imagedata = np.array(imagearray)
    maskdata =  np.array(maskarray)
    # imagedata = np.expand_dims(normalize(imagedata, axis=1), axis=3)
    return imagedata, maskdata


def load_data_nv(root, num_to_read, SIZE = (160, 120, 3), ext = ['tif', 'tif'], allowed_fol = None):
    folders = os.listdir(root)
    # print(folders)
    
    masksList = []
    imagesList = []
    total = 0
    if allowed_fol is None:
      allowed_fol = [i for i in range(1,25)]
    allowed_fol = range(1,9)
    allowed_fol = range(9,18)
    allowed_fol = range(18, 25)
    
    for fol in folders:#in tqdm(folders, desc = "Reading Dataset"):
        
        #if not os.path.isdir(root + "" + fol):
        #  continue
        #if int(fol.split(".")[0]) not in allowed_fol:
        #  continue
        #print(os.path.isdir(root + "" + fol), int(fol.split(".")[0]))
        # print(f"{root}/{fol}/mask-withskin/*.{ext[1]}")
        if os.path.isdir(f"{root}/{fol}") and int(fol) not in allowed_fol:
          continue
        #print(fol)  
        masks = sorted(glob.glob(f"{root}/{fol}/mask-withskin/*.{ext[1]}"))
        images = sorted(glob.glob(f"{root}/{fol}/synthetic/*.{ext[0]}"))
        
        
        #print(len(images), len(masks))
        # print(len(masks))
        #dcdc
    #     total += len(masks)
        for imgName, maskName in (zip(images, masks)):
            img = cv2.imread(imgName)
            mask = cv2.imread(maskName)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if mask is None or img is None:
              continue
            
            total += 1
            if total == num_to_read:
                total = -1
                break
            mask = cv2.imread(maskName, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (SIZE[0], SIZE[1]))
            if SIZE[2] == 1 and img.ndim == 3:
              img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              
            mask = cv2.resize(mask, (SIZE[0], SIZE[1]))

            imagesList.append(img)
            masksList.append(mask)
        if total == -1:
            break
    print(root, total)
    return np.array(imagesList), np.array(masksList)



def read_color_map(file = "classes.json", maskdata = None, wanted = 'pis'):
    dsInfo = readClassesData(fn = file)
    bg_color = dsInfo['maskColor']['Background']
    maskColors =  dsInfo['maskColor']
    
    wanted_classes = replace_(wanted + 'b', {'p':'Pupil', 'i':'Iris', 's': 'Sclera', 'b': 'Background'})
    labels_color = [(c, maskColors[c]) for c in wanted_classes if c in maskColors.keys()]
    unwanted_classes = [(all_classes, maskColors[all_classes]) for all_classes in dsInfo['classes'] if all_classes not in wanted_classes]
    return dsInfo, labels_color, unwanted_classes, wanted_classes



def flatLabels(labels_color, label):
    label_seg = np.zeros(label.shape, dtype=np.uint8)
#     label_seg[np.all(label == Building, axis=-1)] = 0
#     label_seg[np.all(label == Land, axis=-1)] = 1
#     label_seg[np.all(label == road, axis=-1)] = 2
#     label_seg[np.all(label == Vegetation, axis=-1)] = 3
#     label_seg[np.all(label == water, axis=-1)] = 4
#     label_seg[np.all(label == Unlabeled, axis=-1)] = 5
#     label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels
#     return label_seg
    d = {}
    for i, val in enumerate(labels_color):        
        label_seg[np.all(label == val, axis=-1)] = i
        d[i] = val
        
    label_seg = label_seg[:, :, 0]  # Just take the first channel, no need for all 3 channels
    #print("ddddddddddddddddddddddddddddd", d)
    return label_seg, d

def encode_lables_one(maskdata, labels_color1):
    labels_color = list(zip(*labels_color1))[1]
    n_classes = len(labels_color)
    
    y_cat = []

    _, d = flatLabels(labels_color, maskdata[0]) 
    for i in range(maskdata.shape[0]):
        label,_ = flatLabels(labels_color, maskdata[i])    
        y_cat.append(label)




    y_cat = to_categorical(y_cat, num_classes=n_classes)
    return y_cat, d

def encode_lables(y_train, y_val, y_test, maskdata, labels_color1):
    y_cat, d = encode_lables_one(maskdata, labels_color1)    
    y_train_cat, _ = encode_lables_one(y_train, labels_color1)    
    y_val_cat, _ = encode_lables_one(y_val, labels_color1)    
    y_test_cat, _ = encode_lables_one(y_test, labels_color1)    

    return y_train_cat, y_val_cat, y_test_cat, y_cat, d




# def encode_lables(y_train, y_val, y_test, maskdata, labels_color1):
#     labels_color = list(zip(*labels_color1))[1]
#     n_classes = len(labels_color)
    
#     y_train_cat = []
#     y_val_cat = []
#     y_test_cat = []
#     y_cat = []
#     _, d = flatLabels(labels_color, maskdata[0]) 
#     for i in range(maskdata.shape[0]):
#         label,_ = flatLabels(labels_color, maskdata[i])    
#         y_cat.append(label)

#     for i in range(y_train.shape[0]):
#         label,_ = flatLabels(labels_color, y_train[i])
#         y_train_cat.append(label)

#     for i in range(y_val.shape[0]):
#         label,_ = flatLabels(labels_color, y_val[i])
#         y_val_cat.append(label)

#     for i in range(y_test.shape[0]):
#         label,_ = flatLabels(labels_color, y_test[i])
#         y_test_cat.append(label)


#     y_train_cat = to_categorical(y_train_cat, num_classes=n_classes)
#     # y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))  
#     y_test_cat = to_categorical(y_test_cat, num_classes=n_classes)
#     # y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
#     y_val_cat = to_categorical(y_val_cat, num_classes=n_classes)
#     # y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))
#     return y_train_cat, y_val_cat, y_test_cat, y_cat, y_train_cat, y_test_cat, y_val_cat, d


def load_data_all(DATASET, num_to_read, SIZE = (160, 120, 3)):
    print(" DATASET: ", DATASET)
    if DATASET == "NN_human_mouse_eyes":
      DATASET = '../Data/NN_human_mouse_eyes'; imagedata, maskdata = load_data(root = DATASET, num_to_read = num_to_read, SIZE = SIZE)
    elif DATASET == "ClinicAnnotated_DA":
      DATASET = '../Data/ClinicAnnotated_DA'; imagedata, maskdata = load_data(root = DATASET, num_to_read = num_to_read, SIZE = SIZE, ext = ['png', 'png'])
    elif DATASET == "MOBIUS":
      DATASET = '../Data/Eye dataset/pairs'; imagedata, maskdata = load_data1(root = DATASET, num_to_read = num_to_read, SIZE = SIZE)
    elif DATASET == "s-openeds":
      DATASET = r"../Data/s-openeds"; imagedata, maskdata = load_data_nv(root = DATASET, num_to_read = num_to_read, SIZE = SIZE)


    # if DATASET == '../Data/NN_human_mouse_eyes':
    #     imagedata, maskdata = load_data(root = DATASET, num_to_read = num_to_read, SIZE = SIZE)
    # elif DATASET == '../Data/Eye dataset/pairs':
    #     imagedata, maskdata = load_data1(root = DATASET, num_to_read = num_to_read, SIZE = SIZE)
    # elif DATASET == r"../Data/s-openeds":
    #     imagedata, maskdata = load_data_nv(root = DATASET, num_to_read = num_to_read, SIZE = SIZE)
    return imagedata, maskdata


def prepare_data(imagedata, maskdata, testValRatio, testRatio):
    X_train, X_val_test, y_train, y_val_test  = train_test_split(imagedata,
                                                               maskdata,
                                                               test_size = testValRatio,
                                                               random_state=42,
                                                               shuffle = True)

    X_val, X_test, y_val, y_test  = train_test_split(X_val_test,
                                                   y_val_test,
                                                   test_size = testRatio,
                                                   random_state=42,
                                                   shuffle = True)
    x1 = [len(X_train), len(X_val), len(X_test), len(imagedata)]
    x1 = [int(e) for e in x1]
    x2 = np.round(np.array(x1)*100/len(imagedata))
    x2 = [int(e) for e in x2]
    #uts.dataframeAsImage(d = [x1,x2], path = f"{current_model}/module1_results_split.png", rowNames = ["#", "percentage"], colsNames = ["Train", "Validation", "Test", "Total"],  save = True);
    return X_train, y_train, X_val, y_val, X_test, y_test



# import sys 
# sys.path.insert(0, '../Pupil locator/')

SIZE = load_json("../Pupil locator/config.json")['input_size']
SIZE_X, SIZE_Y= SIZE[0], SIZE[1]

print(SIZE)

