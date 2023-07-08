from tensorflow.keras.models import load_model
import models as mnet
from models import jacard_coef, dice_coef, jacard_coef_loss, dice_coef_loss, jacard_coef_loss
from sklearn.utils import class_weight

import cv2
from tensorflow.keras.metrics import MeanIoU
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
import torch
from tqdm import tqdm
import pathlib
import glob
import random
import pandas as pd
from imageProcessing import gamma_correction
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                        white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from skimage import color, morphology
from skimage.segmentation import clear_border
import os 
from sklearn.metrics import mean_squared_error


from models import get_compiled_model
from utilis import load_data_all#(DATASET, num_to_read, SIZE = (128, 128, 3)):



from imageProcessing import gerradius, fill_holes, getCentroid, finCircleContour
from imageProcessing import plot_img, getCentroid, maskHsv,fill_holes, get_objects


def testOnDatasets(DataSetsList, ext,  num = 5, ob = None):
    toread = []
    for idx, ds in enumerate(DataSetsList):
        print(ds)

        d = list(pathlib.Path(ds).rglob(f"*{ext[idx]}"))

        s = random.sample(range(len(d)), num)
        for i in s:
            t = cv2.imread(str(d[i]))
            t  = cv2.resize(t, (128,128))
            toread.append(t)

    for idx, img in enumerate(toread):
        res = ob.predictOneMask(img)
        t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = np.concatenate([t, res], axis = 0)
        if idx == 0:
            total = x
        else:
            total = np.concatenate([total, x], axis = 1)


#     fig,ax = plt.subplots(nrows = 1, figsize=(12,5))
#     plt.imshow(total, cmap="gray")
#     plt.axis('off')
#     plt.savefig("Dataset results")
    return total






class inference:
    def __init__(self, size, n_classes,  BACKBONE, class_weights, model_path = 'weights_best.hdf5'):
        self.size = size
        self.model = get_compiled_model(BACKBONE = BACKBONE, n_classes = n_classes, IMG_SIZE = size, class_weights = class_weights,  model_path = model_path)
    
    
    def predictOnVideo(self, videoPath, split = 1):

        coors = []
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return coors
        f = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                f += 1
                print(f)
                if split == 1:
                    t = []
                    h1, h2 =np.split(frame, 2, axis=1)
                    pred_mask1 = self.predictOneMask(img = h1)
                    pred_mask2 = self.predictOneMask(img = h2)
                    center1 = (2,3)
                    center2 = (2,3)
                    #center1 = getCentroid(pred_mask1)
                    #center2 = getCentroid(pred_mask2)

                    t.extend(center1)
                    t.extend(center2)
                    coors.append(t)
                else:
                    pred_mask = self.predictOneMask(img = frame)
                    center = getCentroid(pred_mask)
                    coors.append(center)
            else: 
                break
        cap.release()
        return coors



    def predictOneMask(self, img):
        img1 = img
        # img[0:20, :] = 255
        # img[:, 0:20] = 255
        # img = cv2.bilateralFilter(img, 5, 5, 5)
        # img = cv2.medianBlur(img, ksize=10)
        
        # # Create the sharpening kernel
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # # Apply the sharpening kernel to the image using filter2D
        
        # img = cv2.filter2D(img, -1, kernel)
        
        # img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # eroded = 255 * (img_gray<30)
        # eroded = closing(eroded, disk(1))
        # center = getCentroid(eroded)
        # ROI = gerradius(img, (70, 70), 50)

        # img1 = ROI


        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, thresh1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + 
        #                                     cv2.THRESH_OTSU)   
        # cv2.imwrite("imgjjj.png", thresh1)


        # img = fill_holes(img)
        # footprint = morphology.disk(2)
        # res1 = img.copy()
        # res2 = img.copy()
        # res1[:, :, 0] = morphology.white_tophat(img[:, :, 0] , footprint)
        # res1[:, :, 1] = morphology.white_tophat(img[:, :, 1] , footprint)
        # res1[:, :, 2] = morphology.white_tophat(img[:, :, 2] , footprint)
        # # cv2.imwrite("w.png", res1)
        # res2[:, :, 0] = morphology.black_tophat(img[:, :, 0] , footprint)
        # res2[:, :, 1] = morphology.black_tophat(img[:, :, 1] , footprint)
        # res2[:, :, 2] = morphology.black_tophat(img[:, :, 2] , footprint)
        # # cv2.imwrite("b.png", res2)
        



        # # res = abs(res - img)
        # res = cv2.bitwise_and(res1, res2)
        # res3 = (cv2.bitwise_xor(res2, res))
        # # cv2.bitwise_not(
        # # cv2.imwrite("img.png", img)
        # img1 = cv2.bitwise_and(res3, img)

        # all_images = np.concatenate([img, res1, res2, res3, img1, res2 + img, cv2.bitwise_not(res3), cv2.bitwise_or(res2 + img, img)], axis = 1)
        # cv2.imwrite("img.png", all_images)
        # img1 = cv2.bitwise_or(res2 + img, img)*1.5

       
        x1 = cv2.resize(img1, (self.size[0], self.size[1]))
        x2 = np.expand_dims(x1, axis=0)
        p1 = self.model.predict(x2, verbose=0)
        p11 = np.argmax(p1, axis = 3)
        # p111  = (np.expand_dims(p11, axis=3))
        p11 = 255*np.squeeze(p11, 0)
        # p11 = clear_border(p11)
        
        # return p11
        # cv2.imwrite("cdcdv.png", fill_holes(img))
        return p11


def testOnAnother(where2save, ob = None):
    
    #"""
    root = "../Data/"
    datasets = [
            [root+"IITD_database/",['.bmp']],
            [root+"MMU-Iris-Database/",['.bmp']],
            [root+"Generated ds/", ['.png']],
            #[root+"s-openeds\\1\\synthetic\\", ['.tif']]
            ]

    
    num = 8
    total = []
    for i, data in enumerate(datasets):
        print(f"inference on some samples from {data[0]}\n")
        print(i)
        res = testOnDatasets(DataSetsList = [data[0]],
            ext = data[1],
            num = num,
            ob = ob)
        if i == 0:
            total = res
        else:
            total = np.concatenate([total, res], axis = 0)

    fig,ax = plt.subplots(nrows = 1, figsize=(10,5))
    plt.imshow(total, cmap="gray")
    plt.axis('off')
    plt.savefig(where2save + "model1_results_datsets_other.png", bbox_inches='tight')

    

    print(f"inference on some samples from {root +'/mineDS/'}\n")
    a = testOnDatasets(DataSetsList = [root +"/mineDS/"],
    ext = ['.bmp'],
    num = 8,
    ob = ob)

    res = np.concatenate([a], axis = 0)
    fig,ax = plt.subplots(nrows = 1, figsize=(10,5))
    plt.imshow(res, cmap="gray")
    plt.axis('off')
    plt.savefig(where2save + "model1_results_datsets_other_mineDS.png", bbox_inches='tight')


    print("#---------------------------------------------------------------------------------\n")
    df = pd.read_csv("../Data/Generated ds/details.csv")
    df['xp'] = ''
    df['yp'] = ''
    print(df.columns)


    for f in tqdm(range(len(df)), desc = "inference on all LPW dataset(MSE metric)"):
        filename, x, y, _, _ = df.iloc[f].values
        filename = int(filename)
        img = cv2.imread(f"../Data/Generated ds/{filename}.png")
        res = ob.predictOneMask(img)
        xp, yp = 0, 0 #getCentroid(res)
        df['xp'].iloc[filename] = xp
        df['yp'].iloc[filename] = yp


    errX = mean_squared_error(df['0'],df['xp'])
    errY = mean_squared_error(df['1'],df['yp'])

    print(f"\n\nerrX: {errX} \nerrY:  {errY}")
    


if __name__ == "__main__":
    # Clearing the Screen
    os.system('cls')

    ob = inference(size = (128, 128, 3), n_classes = 2,  BACKBONE="mine", class_weights = [1,0,0,0], model_path = 'model1_results_segs_best.hdf5')
    testOnAnother(where2save = "", ob = ob)
    # images, masks = readDS(r"C:\Users\Moham\Downloads\Datasets\s-openeds")
    # self.model.evaluate(X_test, y_test_cat, batch_size=128, verbose = 1)












