import pathlib
import os
import cv2
import numpy as np
import glob
import pandas as pd

def saveVideToImages(filename, where, space):
    pathlib.Path(where).mkdir(parents=True, exist_ok=True)

    nFrames = 0
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== True): 
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if nFrames%space == 0:
                    
                    h1 = frame[:, 0:frame.shape[1]//2]
                    h2 = frame[:, frame.shape[1]//2:]
                    cv2.imwrite(f"{where}/{nFrames}_h1.png", h1)
                    cv2.imwrite(f"{where}/{nFrames}_h2.png", h2)
                    #cv2.imwrite(f"{where}/{nFrames}.png", frame)
                nFrames = nFrames + 1
            else: 
                break
        cap.release()
        
        

def saveVideFrames(filename, start, space, where):
    i = start
    nFrames = 0
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    else:
        # Read until video is completed
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if nFrames%space == 0:
                    cv2.imwrite(f"{where}/{i}.png", frame)
                    i += 1
                nFrames = nFrames + 1

            else: 
                break
        # When everything done, release the video capture object
        cap.release()
    return i - start
        
def readLinesFrom(filename, space):
    with open(filename) as f:
        contents = f.read().split("\n")
    linesToRead = list(range(0, len(contents), space))
    tmp = []
    for l in linesToRead:
        lineData = contents[l]
        if lineData != '':
            t = lineData.split()
            tmp.append(t)
    return tmp

from tqdm import tqdm

def saveVideoTxtDS_images(root, where, space):
    pathlib.Path(where).mkdir(parents=True, exist_ok=True)

    videoFiles = sorted(list(pathlib.Path(root).rglob("*.avi")))
    videoLabels = sorted(list(pathlib.Path(root).rglob("*.txt")))

    data = zip(videoFiles, videoLabels)

    i = 0
    nVideo = 0
    ds = []
    for videoName, lbl in tqdm(data, total = len(videoFiles)):
        nSavedFrames = saveVideFrames(str(videoName), i, space = space, where = where)
        i = i + nSavedFrames

        d = readLinesFrom(lbl, space = space)
        ds.extend(d)
        nVideo += 1
    #     if nVideo == 3:
    #         break

    df = pd.DataFrame(ds)
    df.to_csv(f"{where}/details.csv")
    
if __name__ == "__main__":
    saveVideoTxtDS_images(root = r"C:\Users\Moham\Downloads\archive\LPW", where = "../Data/LPW", space = 1000)