import pathlib
import os
import cv2
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def replace(img_rgb, pixel1, pixel2):
    
    r1, g1, b1 = pixel1
    r2, g2, b2 = pixel2
    red, green, blue = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    img_rgb[:,:,:4][mask] = [r2, g2, b2]
#     im = Image.fromarray(img_rgb)
    return img_rgb


    
if __name__ == "__main__":


    #--------------------------------------------------------------------------





    #--------------------------------------------------------------------------
    color_system = [
                    [[255, 255, 255], [0, 0, 255]],
    ]
    root = "../Data/ClinicAnnotated"
    Images = sorted(glob.glob(f"{root}/Images/*.bmp"))
    # Masks = sorted(glob.glob(f"{root}/Masks/*.png"))
    # print(Images)
    pathlib.Path(root + "/Masks").mkdir(parents=True, exist_ok=True)
    for fn in Images:
        name = Path(fn).name.split('.')[0] + '.tiff'
        maskName = fr"{root}/Masks1/{name}"
        # print(maskName)
        img = cv2.imread(maskName)

        img = img * 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = img.copy()
        for c in color_system:
            img1 = replace(img1, c[0], c[1])
        print(f"{root}/Masks/{name}")
        cv2.imwrite(f"{root}/Masks/{name}", img1)

