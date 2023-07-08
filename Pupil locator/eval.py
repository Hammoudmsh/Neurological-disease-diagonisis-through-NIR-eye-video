
from parsing_file import create_parser_eval
from tqdm import tqdm
# from pathlib import Path
import pathlib
import datetime
from utilis import load_data, load_data_all,load_data1, load_data_nv, ClearToBlack, encode_lables_one, read_color_map,append2csv

from models import get_compiled_model

import cv2
import numpy as np

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
#from skimage.morphology import (erosion, dilation, opening, closing, white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk  # noqa
from skimage import color, morphology
from skimage.segmentation import clear_border
import os 
from sklearn.metrics import mean_squared_error


from models import get_compiled_model
from utilis import load_data_all, to3D#(DATASET, num_to_read, SIZE = (128, 128, 3)):



from imageProcessing import gerradius, fill_holes, getCentroidAndArea, finCircleContour
from imageProcessing import plot_img, getCentroidAndArea, maskHsv,fill_holes, get_objects

# import inference
# from inference import inference

#--------------------------------------------------------------------------------------------------------



    

class inference:
    def __init__(self, size, n_classes,  BACKBONE, class_weights, model_path = 'weights_best.hdf5'):
        self.size = size
        self.model = get_compiled_model(BACKBONE = BACKBONE, n_classes = n_classes, IMG_SIZE = size, class_weights = class_weights,  model_path = model_path)
    
    # def predictOnVideo(self, videoPath):

    #     coors = []
    #     cap = cv2.VideoCapture(videoPath)
    #     if (cap.isOpened()== False): 
    #         print("Error opening video stream or file")
    #         return coors
    #     f = 0
    #     while(cap.isOpened()):
    #         ret, frame = cap.read()
    #         if ret == True:
    #             pred_mask1 = self.predictOneMask(img = frame)
    #             pred_mask1 = np.dstack((pred_mask1, pred_mask1, pred_mask1))
    #             print(pred_mask1.shape)
    #             pred_mask1 = cv2.resize(pred_mask1, (320, 120))
    #             cv2.imwrite(f"zz/{f}.png", pred_mask1)
    #             f += 1
    #         else: 
    #             break
    #     cap.release()
    #     return coors

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

        # print(self.size)
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
def readEyerecordingAsframes(videoPath, SIZE, CLANE, num_frames, split):
    h1_all = []
    h2_all = []
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return [], []
    f = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if split:
                h1, h2 =np.split(frame, 2, axis=1)
                h1 = cv2.resize(h1, (SIZE[0], SIZE[1]))
                h2 = cv2.resize(h2, (SIZE[0], SIZE[1]))
                if CLANE:
                    h1 = get_CLANE(h1)
                    h2 = get_CLANE(h2)
                h1_all.append(h1)
                h2_all.append(h2)
            else:
                frame = cv2.resize(frame, (SIZE[0], SIZE[1]))
                if CLANE:
                    frame = get_CLANE(frame)
                h1_all.append(frame)
            f += 1
        # if f%50 == 0:
        #     print(f)
        if f == num_frames:
            break 
    return np.array(h1_all), np.array(h2_all)

def writeVideo(images, video_name, fs = 50):
    height, width, layers = images[0].shape    
    video = cv2.VideoWriter(video_name, 0, fs, (width,height))

    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()

def write_video(file_path, frames, fps = 50):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h, _ = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, 0, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release() 

def getNeededInfo(ob, img):
    pred = ob.predictOneMask(img)
    pred = to3D(pred).astype("uint8")*255
    props, gaze, mask, cnt = getCentroidAndArea(pred)
    if props is  None:
        mask = gaze = img
        cnt = []
        props = {"centroid-0": [-1],"centroid-1": [-1],"centroid-1": [-1], "axis_major_length": [-1], "axis_minor_length":[-1], "area":-1, "cX":-1, "cY":-1}
    return pred, props, gaze, mask, cnt


def predictOnEyeVideo(ob, videoPath, saveTo = '', CLANE = 1, num_frames = -1):    
    SIZE = ob.size
    features_left = []
    features_right = []
    video_images = []
    cap = cv2.VideoCapture(videoPath)
    
    if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            return [], []
    f = 0
    while(cap.isOpened()):
        print("frame:", f)

        ret, frame = cap.read()
        if ret == True:
            h1, h2 =np.split(frame, 2, axis=1)
            h1 = cv2.resize(h1, (SIZE[0], SIZE[1]))
            h2 = cv2.resize(h2, (SIZE[0], SIZE[1]))
            if CLANE:
                h1 = get_CLANE(h1)
                h2 = get_CLANE(h2)

            pred1, props1, gaze1, mask1, cnt1 = getNeededInfo(ob, h1)
            pred2, props2, gaze2, mask2, cnt2 = getNeededInfo(ob, h2)
            print(props1)
            features_left.append(props1)
            features_right.append(props2)
        if 0:#saveTo != '':

            org_frame = np.concatenate([h1, h2], axis = 1) 
            seg_frame = np.concatenate([to3D(pred1)*255, to3D(pred2)*255], axis = 1)
            h11 = h1.copy()
            h22 = h2.copy()

            if len(cnt1) != 0:
                h11 = cv2.ellipse(h1.copy(), cnt1, (0,0,255), 1)
            if len(cnt2) != 0:
                h22 = cv2.ellipse(h2.copy(), cnt2, (0,0,255), 1)
            gaze_frame = np.concatenate([h11, h22], axis = 1)

            # gaze_frame = np.concatenate([gaze1[i], gaze2[i]], axis = 1)
            # print(org_frame.shape, seg_frame.shape, gaze_frame.shape)
            all_parts = np.concatenate([org_frame, seg_frame, gaze_frame], axis = 0)
            cv2.imwrite(f"{saveTo}/{f}.png", all_parts)
            video_images.append(gaze_frame)
        f += 1
        if f == num_frames:
            break 
    if 0:#saveTo != '':

        writeVideo(video_images, "res.avi", fs = cap.get(cv2.CAP_PROP_FPS))

    features_left = pd.DataFrame.from_dict(features_left)
    features_right = pd.DataFrame.from_dict(features_right)
    
    features_left = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})
    features_right = features_right.rename(columns={"centroid-0": "c0_right", "centroid-1": "c1_right", "centroid-2":"c2_right", "axis_major_length":"axis_major_right", "axis_minor_length":"axis_minor_right", "left": "area_right", "cX":"cX_right", "cY":"cY_right"}) 
    return 1, pd.concat([features_left, features_right], axis=1)

    # # write_video("resVideo.mp4", video_images)
    # features_left = pd.DataFrame.from_dict(features_left)
    # features_right = pd.DataFrame.from_dict(features_right)
    
    # features_left = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})
    # features_right = features_right.rename(columns={"centroid-0": "c0_right", "centroid-1": "c1_right", "centroid-2":"c2_right", "axis_major_length":"axis_major_right", "axis_minor_length":"axis_minor_right", "left": "area_right", "cX":"cX_right", "cY":"cY_right"}) 
    # return 1, pd.concat([features_left, features_right], axis=1)
    


def predictOnEyeVideo1(ob, videoPath, saveTo = '', CLANE = 1.0, num_frames = -1):    
    SIZE = ob.size
    features_left = []
    features_right = []
    
    all_h1, all_h2 = readEyerecordingAsframes(videoPath, SIZE, CLANE, num_frames = num_frames, split = 1)
    if len(all_h1) == 0:
        return 0, []

    
    preds1, features_left, gaze1, masks1, cnt1 = predictOnBatch(ob, all_h1)
    print("_-----------")
    preds2, features_right, gaze2, masks2, cnt2 = predictOnBatch(ob, all_h2)
    video_images = []
    if saveTo != '':
        for i in range(len(all_h1)):
            org_frame = np.concatenate([all_h1[i], all_h2[i]], axis = 1) 
            seg_frame = np.concatenate([to3D(preds1[i])*255, to3D(preds2[i])*255], axis = 1)
            
            h11 = cv2.ellipse(all_h1[i].copy(), cnt1[i], (0,0,255), 1)
            h22 = cv2.ellipse(all_h2[i].copy(), cnt2[i], (0,0,255), 1)
            gaze_frame = np.concatenate([h11, h22], axis = 1)

            # gaze_frame = np.concatenate([gaze1[i], gaze2[i]], axis = 1)
            # print(org_frame.shape, seg_frame.shape, gaze_frame.shape)
            all_parts = np.concatenate([org_frame, seg_frame, gaze_frame], axis = 0)
            cv2.imwrite(f"{saveTo}/{i}.png", all_parts)
            video_images.append(gaze_frame)
    writeVideo(video_images, "res.avi")
    # write_video("resVideo.mp4", video_images)
    features_left = pd.DataFrame.from_dict(features_left)
    features_right = pd.DataFrame.from_dict(features_right)
    
    features_left = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})
    features_right = features_right.rename(columns={"centroid-0": "c0_right", "centroid-1": "c1_right", "centroid-2":"c2_right", "axis_major_length":"axis_major_right", "axis_minor_length":"axis_minor_right", "left": "area_right", "cX":"cX_right", "cY":"cY_right"}) 
    return 1, pd.concat([features_left, features_right], axis=1)
    
def predictOnVideo(ob, videoPath, saveTo = '', CLANE = 1.0, num_frames = -1):    
    SIZE = ob.size
    all_h1, _ = readEyerecordingAsframes(videoPath, SIZE, CLANE, num_frames = num_frames, split= False)
    if len(all_h1) == 0:
        return 0, []

    
    preds1, features_left, gaze1, masks1, cnt1 = predictOnBatch(ob, all_h1)
    video_images = []
    if saveTo != '':
        for i in range(len(all_h1)):
            org_frame =  all_h1[i]
            seg_frame = to3D(preds1[i])*255            
            h11 = cv2.ellipse(all_h1[i].copy(), cnt1[i], (0,0,255), 1)
            gaze_frame = h11
            # gaze_frame = np.concatenate([gaze1[i], gaze2[i]], axis = 1)
            # print(org_frame.shape, seg_frame.shape, gaze_frame.shape)
            all_parts = np.concatenate([org_frame, seg_frame, gaze_frame], axis = 0)
            cv2.imwrite(f"{saveTo}/{i}.png", all_parts)
            video_images.append(gaze_frame)
    writeVideo(video_images, "res.avi")
    # write_video("resVideo.mp4", video_images)
    features_left = pd.DataFrame.from_dict(features_left)
    
    features_left = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})
    return 1, features_left
    
# def predictOnEyeVideo1(ob, videoPath, saveTo = ''):
    
#     SIZE = ob.size
#     features_left = []
#     features_right = []
    
#     all_h1, all_h2 = readEyerecordingAsframes(videoPath, SIZE)
#     if len(all_h1) == 0:
#         return 0, []
#     preds1 = ob.model.predict(all_h1, verbose=1)#, batch_size = 128)
#     preds2 = ob.model.predict(all_h2, verbose=1)#, batch_size = 128)
    
#     preds1 = np.argmax(preds1, axis = 3)
#     preds2 = np.argmax(preds2, axis = 3)
# #     plt.figure(), plt.imshow(preds1[0], cmap = "gray")
#     for f, (pred1, pred2) in enumerate(zip(preds1, preds2)):
#         pred1 = to3D(pred1).astype("uint8")*255
#         pred2 = to3D(pred2).astype("uint8")*255
# #         pred1 = cv2.cvtColor(pred1, cv2.COLOR_BGR2RGB)
# #         pred2 = cv2.cvtColor(pred2, cv2.COLOR_BGR2RGB)
# #         plt.figure(), plt.imshow(pred1*255, cmap = "gray")
# #         vdvdvd
#         props1, gaze1, mask1, cnt1 = getCentroidAndArea(pred1)
#         props2, gaze2, mask2, cnt2 = getCentroidAndArea(pred2)
# #         plt.imshow(gaze1)
# #         ccdv
#         if saveTo != '':
#             h11 = cv2.ellipse(all_h1[f].copy(), cnt1, (0,0,255), 1)
#             h22 = cv2.ellipse(all_h2[f].copy(), cnt2, (0,0,255), 1)
#             # org_frame = np.concatenate([all_h1[f], all_h2[f]], axis = 1)            
#             # seg_frame = np.concatenate([pred1, pred2], axis = 1)
#             # org_frame1 = np.concatenate([h11, h22], axis = 1)            

#             # gaze = np.concatenate([gaze1, gaze2], axis = 1)
#             # resss = np.concatenate([to3D(org_frame), org_frame1, seg_frame, gaze], axis = 0)

#             resss = np.concatenate([all_h1[f], pred1, h11], axis = 0)

#             cv2.imwrite(f"{saveTo}/{f}.png", resss)
#         props2["frame_id"] = f
#         features_left.append(props1)
#         features_right.append(props2)
#     features_left = pd.DataFrame.from_dict(features_left)
#     features_right = pd.DataFrame.from_dict(features_right)
#     features_left = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})
#     features_right = features_right.rename(columns={"centroid-0": "c0_right", "centroid-1": "c1_right", "centroid-2":"c2_right", "axis_major_length":"axis_major_right", "axis_minor_length":"axis_minor_right", "left": "area_right", "cX":"cX_right", "cY":"cY_right"}) 
#     return 1, pd.concat([features_left, features_right], axis=1)
    
# def predictOnEyeVideo(ob, videoPath, split = 1, saveTo = ''):
#     SIZE = ob.size
#     features_left = []
#     features_right = []
#     cap = cv2.VideoCapture(videoPath)
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")
#         return coors
#     f = 0
    
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if f%15 == 0:
#             print(f)
#         if ret == True:
#             if split == 1:
#                 h1, h2 =np.split(frame, 2, axis=1)
#                 pred_mask1 = ob.predictOneMask(img = h1)
#                 pred_mask2 = ob.predictOneMask(img = h2)
#                 pred_mask1 = to3D(pred_mask1).astype("uint8")
#                 pred_mask2 = to3D(pred_mask2).astype("uint8")

# #                 pred_mask1 = cv2.cvtColor(pred_mask1, cv2.COLOR_BGR2RGB)
# #                 pred_mask2 = cv2.cvtColor(pred_mask2, cv2.COLOR_BGR2RGB)

# #                 print(h1.shape, pred_mask1.shape)

#                 props1, gaze1, mask1 = getCentroidAndArea(pred_mask1)
#                 props2, gaze2, mask2 = getCentroidAndArea(pred_mask2)
#                 props2["id"] = f

#                 features_left.append(props1)
#                 features_right.append(props2)


#                 if saveTo != '':
#                     seg_frame = np.concatenate([pred_mask1, pred_mask2], axis = 1)
#                     gaze = np.concatenate([to3D(gaze1, SIZE), to3D(gaze2, SIZE)], axis = 1)
#                     resss = np.concatenate([to3D(seg_frame), gaze], axis = 0)
#                     cv2.imwrite(f"{saveTo}/{f}.png", seg_frame)
#             f += 1
#         else: 
#             break
#     cap.release()
#     features_left = pd.DataFrame.from_dict(features_left)
#     features_right = pd.DataFrame.from_dict(features_right)
#     features_left = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})
#     features_right = features_right.rename(columns={"centroid-0": "c0_right", "centroid-1": "c1_right", "centroid-2":"c2_right", "axis_major_length":"axis_major_right", "axis_minor_length":"axis_minor_right", "left": "area_right", "cX":"cX_right", "cY":"cY_right"})
    
#     return pd.concat([features_left, features_right], axis=1)

# def testOnDatasets(DataSetsList, ext,  num = 5, ob = None):
#     toread = []
#     for idx, ds in enumerate(DataSetsList):
#         print(ds)

#         d = list(pathlib.Path(ds).rglob(f"*{ext[idx]}"))

#         s = random.sample(range(len(d)), num)
#         for i in s:
#             t = cv2.imread(str(d[i]))
#             t  = cv2.resize(t, (160,120))
#             toread.append(t)

#     for idx, img in enumerate(toread):
#         res = ob.predictOneMask(img)
#         t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         x = np.concatenate([t, res], axis = 0)
#         if idx == 0:
#             total = x
#         else:
#             total = np.concatenate([total, x], axis = 1)


# #     fig,ax = plt.subplots(nrows = 1, figsize=(12,5))
# #     plt.imshow(total, cmap="gray")
# #     plt.axis('off')
# #     plt.savefig("Dataset results")
#     return total





'''
def testOnAnother(where2save, ob = None):
    
    #"""
    root = "../Data/"
    datasets = [
            [root+"IITD_database/",['.bmp']],
            [root+"MMU-Iris-Database/",['.bmp']],
            [root+"LPW/", ['.png']],
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
    df = pd.read_csv("../Data/LPW/details.csv")
    df['xp'] = ''
    df['yp'] = ''
    print(df.columns)


    for f in tqdm(range(len(df)), desc = "inference on all LPW dataset(MSE metric)"):
        filename, x, y, _, _ = df.iloc[f].values
        filename = int(filename)
        img = cv2.imread(f"../Data/LPW/{filename}.png")
        res = ob.predictOneMask(img)
        xp, yp = 0, 0 #getCentroid(res)
        df['xp'].iloc[filename] = xp
        df['yp'].iloc[filename] = yp


    errX = mean_squared_error(df['0'],df['xp'])
    errY = mean_squared_error(df['1'],df['yp'])

    print(f"\n\nerrX: {errX} \nerrY:  {errY}")
'''

def plot_res(imagedata, maskdata = None, num = 8):       
    

    s = random.sample(range(len(imagedata)), num)
    
    for idx, i in enumerate(s):
        img = imagedata[i]
        
        if maskdata is not None:
            mask = maskdata[i]

            row = np.concatenate([to3D(img), to3D(mask)*255], axis = 0)
        else:
            row = np.concatenate([to3D(img)], axis = 0)

        if idx == 0:
            total = row
        else:
            total = np.concatenate([total,row], axis = 1)
    return total

def predictOnSome(imagedata, maskdata = None, num = 8, ob1 =  None):       
    
        
    if ob1 is None:
      global ob
      ob1 = ob

    s = random.sample(range(len(imagedata)), num)
    
    for idx, i in enumerate(s):

        pred = ob1.predictOneMask(imagedata[i])
        img = cv2.cvtColor(imagedata[i], cv2.COLOR_BGR2RGB)

        if maskdata is not None:
            mask = maskdata[i]
            row = np.concatenate([to3D(img), to3D(mask), to3D(pred)], axis = 0)
        else:
            row = np.concatenate([to3D(img), to3D(pred)], axis = 0)

        if idx == 0:
            total = row
        else:
            total = np.concatenate([total,row], axis = 1)
    return total


def predictOnBatch(ob, all_images):
  masks = []
  gazes = []
  features = []
  cnts = []
  preds = []
  for i, img in enumerate(all_images):
    # print(i)
    pred, props, gaze, mask, cnt = getNeededInfo(ob, img)
    preds.append(pred)
    gazes.append(gaze)
    masks.append(mask)
    features.append(props)
    cnts.append(cnt)
  return preds, features, gazes, masks, cnts
    

def predictOnBatch1(ob, all_images):
  masks = []
  gazes = []
  features = []
  cnts = []
  preds = ob.model.predict(all_images, verbose=1, batch_size = 16)
  preds = np.argmax(preds, axis = 3)
  for f, pred in enumerate(preds):
    pred = to3D(pred).astype("uint8")*255
    props, gaze, mask, cnt = getCentroidAndArea(pred)
    if props is  None:
        mask = gaze = all_images[f]
        cnt = []
        props = {"centroid-0": [-1],"centroid-1": [''],"centroid-1": [''], "axis_major_length": [''], "axis_minor_length":[''], "area":'', "cX":'', "cY":''}

    gazes.append(gaze)
    masks.append(mask)
    features.append(props)
    cnts.append(cnt)
  return preds, features, gazes, masks, cnts
    

def cleanIfNeeded(*var):
    out_list = []
    for i in var:
        if isinstance(i, str):
            out_list.append(i.strip("''"))
        else:
            out_list.append(var)
    return out_list


def get_CLANE(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img[:,:,1] = clahe.apply(img[:,:,1])
    img[:,:,2] = clahe.apply(img[:,:,2])
    return img

def process(images, CLANE = False):
    if CLANE:    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        images1 = []
        for i, img in enumerate(images):
            img[:,:,0] = clahe.apply(img[:,:,0])
            img[:,:,1] = clahe.apply(img[:,:,1])
            img[:,:,2] = clahe.apply(img[:,:,2])
            images1.append(img)

    return images1


from collections import Counter

def eval():
    global ob

    root = "../Data/"


    parser = create_parser_eval()
    args = parser.parse_args()

    wanted, out, MODEL_PATH, imgExt, maskExt = cleanIfNeeded(args.WANTED, args.OUTPUT_PATH, args.MODEL_PATH, args.imgExt, args.maskExt)
    # wanted = args.WANTED#'pis'
    # out = args.OUTPUT_PATH
    # MODEL_PATH = args.MODEL_PATH
    # imgExt = args.imgExt
    # maskExt = args.maskExt

    n_classes = args.CLASSES_NUM#2
    CLANE = args.CLANE

    if '.hdf5' not in MODEL_PATH:
        MODEL_PATH = MODEL_PATH + '.hdf5'


    num = 8
    num_to_read = 0
    class_weights = [1,0,0,0]

    from utilis import load_json
    SIZE = load_json("config.json")['input_size']
    SIZE_X, SIZE_Y= SIZE[0], SIZE[1]
    BATCH_SIZE_EVALUATE = load_json("config.json")['BS_eval']



    #print("-------------------Info-----------------")
    #print("\t#classes:", n_classes)
    #print("\twanted:", wanted)
    #print("\tSIZE:", SIZE)
    #print("\tBATCH_SIZE_EVALUATE:", BATCH_SIZE_EVALUATE)
    #print("\tclass_weights:", class_weights)
    
    #print("\model_path:", MODEL_PATH)


    ob = inference(size = SIZE, n_classes = n_classes,  BACKBONE="mine", class_weights = class_weights, model_path = MODEL_PATH)
    # model = get_compiled_model(BACKBONE = "mine", n_classes = n_classes, IMG_SIZE = SIZE, class_weights = class_weights,  model_path = MODEL_PATH)


    
    


    if args.DS_NAME != '':
        print(f"-----------------------------------------evaluate {args.DS_NAME}")
        #print(args.SEGMENT)
        if args.SEGMENT:
            # imagedata, maskdata = load_data_all(root = args.DS_NAME, num_to_read = 0, SIZE = SIZE)
            if args.DS_NAME == "NN_human_mouse_eyes":
                # ext = '.bmp'
                DATASET = '../Data/NN_human_mouse_eyes'; imagedata, maskdata = load_data(root = DATASET, num_to_read = num_to_read, SIZE = SIZE, ext = [imgExt, maskExt])
            elif args.DS_NAME == "ClinicAnnotated_DA":
                # ext = '.bmp'
                DATASET = '../Data/ClinicAnnotated_DA'; imagedata, maskdata = load_data(root = DATASET, num_to_read = num_to_read, SIZE = SIZE, ext = [imgExt, maskExt])

            elif args.DS_NAME == "MOBIUS":
                # ext = '.bmp'
                DATASET = '../Data/Eye dataset/pairs'; imagedata, maskdata = load_data1(root = DATASET, num_to_read = num_to_read, SIZE = SIZE, ext = [imgExt, maskExt])
            elif args.DS_NAME in ["s-openeds", "s-nvgaze", "s-natural"]:       
                DATASET = rf"../Data/{args.DS_NAME}"; imagedata, maskdata = load_data_nv(root = DATASET, num_to_read = num_to_read, SIZE = SIZE, ext = [imgExt, maskExt])
            
            
            
                  
            imagedata = imagedata
            
            
            dsInfo, labels_color, unwanted, wanted_classes = read_color_map(file = DATASET + "/classes.json", maskdata = maskdata,  wanted = wanted)
            print("______________________________________", labels_color)
            n_classes = len(labels_color)
            bg_color = np.array(dsInfo['maskColor']['Background'])

            # print("unwanted: ", unwanted)
            before = maskdata[0]
            for i, mask in enumerate(maskdata):
                for uw in unwanted:
                    maskdata[i] = ClearToBlack(mask, list(uw)[1])
            # print("_____________________")
            xxx = np.concatenate([before, maskdata[0]])
            cv2.imwrite(f"check_{args.DS_NAME}.png", xxx)

            maskdata1 = encode_lables_one(maskdata, labels_color)[0]
            
            res = predictOnSome(imagedata, maskdata, num = 8)
            cv2.imwrite(f"{out}res_{args.DS_NAME}.png", res)            
            print("******************", imagedata.shape)            
            
            imagedata_ = np.array_split(imagedata, len(imagedata)//5)
         
            maskdata1_ = np.array_split(maskdata1, len(maskdata1)//5)
            
            del imagedata
            del maskdata1
            results = {}
            parts_num = 0
            for idx, (imagedata_part, maskdata1_part) in enumerate(zip(imagedata_, maskdata1_)):#, desc = "parts", total = len(imagedata_)):
              results_tmp = ob.model.evaluate(imagedata_part, maskdata1_part, batch_size=BATCH_SIZE_EVALUATE, verbose = 0, return_dict=True, use_multiprocessing=True)
              #print(results)
              results = dict(Counter(results)+Counter(results_tmp))
              parts_num += 1
            
              del imagedata_[idx]
              del maskdata1_[idx]
            #print(results)
            #results = ob.model.evaluate(imagedata, maskdata1, batch_size=BATCH_SIZE_EVALUATE, verbose = 0, return_dict=True, use_multiprocessing=True)



            #print(ob.model.metrics_names)
            for k, v in results.items():
              results[k] = [np.round(v, 5)/parts_num]
                
            #results = dict(zip(ob.metrics_names,results))
            
            
            results["dataset_name"] = [args.DS_NAME]
            print("____________________________________________________________", f"{out}evaluation_results_all.csv")
            
            print("**********************************************")
            print(results)
            print("**********************************************")
            df = pd.DataFrame.from_dict(results)
            order = ['dataset_name', 'mean_io_u', 'iou_pupil', 'iou_bg', 'iou_mean1', 'iou_mean2', 'f1-score', 'loss', 'precision', 'recall', 'dice_coef', 'jacard_coef', 'iou_score', 'auc']
            append2csv(f"{out}evaluation_results_all.csv", df[order])

        else:
            if args.DS_NAME == "IITD_database":
                DATASET = '../Data/IITD_database';
            elif args.DS_NAME == "MMU-Iris-Database":
                DATASET = '../Data/MMU-Iris-Database';
            elif args.DS_NAME == "LPW":
                DATASET = '../Data/LPW';

            imagesPath = list(pathlib.Path(DATASET+"/").rglob(f"*.{imgExt}"))

            all_images = []
            for fn in imagesPath:
                img = cv2.imread(str(fn))
                img = cv2.resize(img, (SIZE_X, SIZE_Y))
                all_images.append(img)
            all_images = np.array(all_images)
            
            preds, features, gazes, masks, cnts = predictOnBatch(ob, all_images)
            res = plot_res(all_images, preds, num = 8)
            cv2.imwrite(f"{out}eval_{args.DS_NAME}.png", res)
            

    elif args.IMG_PATH != '':

        out_path = str(pathlib.Path(out).parent)
        out_fn = str(pathlib.Path(out).name)

        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
        total = []

        all_images = []
        for fn in args.IMG_PATH:
            img = cv2.imread(fn)
            img = cv2.resize(img, (SIZE_X, SIZE_Y))
            all_images.append(img)

        all_images = process(all_images, CLANE)
        all_images = np.array(all_images)
        
        preds, features, gazes, masks, cnts = predictOnBatch(ob, all_images)
        features = pd.DataFrame.from_dict(features)
        # features = features_left.rename(columns={"centroid-0": "c0_left", "centroid-1": "c1_left", "centroid-2":"c2_left", "axis_major_length":"axis_major_left", "axis_minor_length":"axis_minor_left", "left": "area_left", "cX":"cX_left", "cY":"cY_left"})

        for i in range(len(all_images)):
            gaze_frame = all_images[i].copy()
            name = pathlib.Path(args.IMG_PATH[i]).name
            print(cnts[i])

            if len(cnts[i]) != 0:
                gaze_frame = cv2.ellipse(all_images[i].copy(), cnts[i], (0,0,255), 1)
            all_parts = np.concatenate([all_images[i], to3D(preds[i])*255, gaze_frame], axis = 0)
            cv2.imwrite(f"{out_path}/{name}", all_parts)      
            features.iloc[i].to_csv(f"{out_path}/{name.split('.')[0]}.csv")
            total.append(all_parts)
        total = np.concatenate(total, axis = 1)        
        cv2.imwrite(f"{out_path}/{out_fn}", total)



    # elif args.IMG_PATH != '':

    #     out_path = str(pathlib.Path(out).parent)
    #     out_fn = str(pathlib.Path(out).name)

    #     pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    #     total = []
    #     print(args.IMG_PATH)
        
        
    #     for fn in args.IMG_PATH:
    #         img = cv2.imread(fn)
    #         name = pathlib.Path(fn).name
    #         img = cv2.resize(img, (SIZE_X, SIZE_Y))
    #         pred1 = ob.predictOneMask(img.copy())
    #         pred1 = to3D(pred1).astype("uint8")#*255
    #         props1, gaze1, mask1, cnt1 = getCentroidAndArea(pred1)
    #         h11 = cv2.ellipse(img.copy(), cnt1, (0,0,255), 1)

    #         resss = np.concatenate([img, pred1, h11], axis = 0)        
    #         cv2.imwrite(f"{out_path}/{name}", resss)
    #         features = pd.DataFrame.from_dict(props1)
    #         features.to_csv(f"{out_path}/{name.split('.')[0]}.csv")
    #         total.append(resss)
    #     total = np.concatenate(total, axis = 1)        
    #     cv2.imwrite(f"{out_path}/{out_fn}", total)
    elif args.VIDEO_PATH != '':
        import time
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        st = time.time()

        success, features = predictOnEyeVideo(ob,  args.VIDEO_PATH, saveTo = out, CLANE = False, num_frames = 10)
        print(time.time() - st)
        # success, features = predictOnVideo(ob,  args.VIDEO_PATH, saveTo = out, CLANE =CLANE, num_frames = 500)

        if success:
            features.to_csv(f"{out}features.csv")

    # except:
    #     print("error")

if __name__ == '__main__':
    eval()