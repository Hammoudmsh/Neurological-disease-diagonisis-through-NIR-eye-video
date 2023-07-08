import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import pathlib
from tqdm import tqdm
import glob
import albumentations as A
import cv2



if __name__ == "__main__":
    #SIZE = (160, 120)
    from utilis import load_json
    SIZE = load_json("config.json")['input_size']
    SIZE_X, SIZE_Y= SIZE[0], SIZE[1]

    images_to_generate  = 4000
    images_path="../Data/ClinicAnnotated/Images/" #path to original images
    masks_path = "../Data/ClinicAnnotated/Masks/"

    img_augmented_path="../Data/ClinicAnnotated_DA/Images/" # path to store aumented images
    msk_augmented_path="../Data/ClinicAnnotated_DA/Masks/" # path to store aumented images

    pathlib.Path(img_augmented_path).mkdir(parents=True, exist_ok=True)#metrics
    pathlib.Path(msk_augmented_path).mkdir(parents=True, exist_ok=True)#metrics

    images = glob.glob(f"{images_path}*.png")
    masks = glob.glob(f"{masks_path}*.tiff")

    aug = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=1),
        A.Transpose(p=1),
        #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=1)
        ]
    )




    for i in tqdm(range(images_to_generate), desc = "generate images", ncols = 100):
        number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
        image = images[number]
        mask = masks[number]
    #     print(image, mask)
        #image=random.choice(images) #Randomly select an image name
      
        original_image = cv2.imread(image)
        original_mask = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)

        
        augmented = aug(image=original_image, mask=original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']

            
        # new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
        # new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)

        new_image_path= "%s/%s.png" %(img_augmented_path, i)
        new_mask_path = "%s/%s.png" %(msk_augmented_path, i)
        
        transformed_image = cv2.resize(transformed_image, (SIZE_X, SIZE_Y))
        transformed_mask = cv2.resize(transformed_mask, (SIZE_X, SIZE_Y))

        io.imsave(new_image_path, transformed_image)
        io.imsave(new_mask_path, transformed_mask)
        if i==4:
            break;

import shutil
shutil.copy2('../Data/ClinicAnnotated/classes.json', '../Data/ClinicAnnotated_DA/classes.json') # target filename is /dst/dir/file.ext



