import pathlib
import os
import cv2
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm


def saveVideToImages(filename, where, space):
    pathlib.Path(where).mkdir(parents=True, exist_ok=True)
    name = pathlib.Path(filename).name.split(".")[0]
    nFrames = 0
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== True): 
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if nFrames%space == 0:                    
                    h1 = frame[:, 0:frame.shape[1]//2]
                    h2 = frame[:, frame.shape[1]//2:]
                    h1 = cv2.resize(h1, (320, 320))
                    h2 = cv2.resize(h2, (320, 320))
                    cv2.imwrite(f"{where}/{name}_{nFrames}_h1.bmp", h1)
                    cv2.imwrite(f"{where}/{name}_{nFrames}_h2.bmp", h2)
                nFrames = nFrames + 1
            else: 
                break
        cap.release()
        

def saveVideoTxtDS_images(root, where, space):
    pathlib.Path(where).mkdir(parents=True, exist_ok=True)

    videoFiles = sorted(list(pathlib.Path(root).rglob("*.avi")))


    for videoName in tqdm(videoFiles, total = len(videoFiles)):
        saveVideToImages(str(videoName), space = space, where = where)

    
if __name__ == "__main__":
    saveVideoTxtDS_images(root = r"../Data/DataSet", where = "../Data/mineDS", space = 1000)