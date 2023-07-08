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



import pathlib
import glob
import random





def get_model(name = "mine", n_classes = 2, IMG_SIZE = (256, 256, 256)):
    return mnet.get_model(name = name, n_classes = n_classes, IMG_SIZE = IMG_SIZE)



def testOnDatasets(DataSetsList, ext,  num = 5):
    toread = []
    
    for idx, ds in enumerate(DataSetsList):
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
        self.model = model = get_model(name = BACKBONE, n_classes = n_classes, IMG_SIZE = size)
        
        my_metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
            jacard_coef,
        #     tf.keras.metrics.MeanIoU(num_classes = n_classes),
        #     tf.keras.metrics.TruePositives(name='tp'),
        #     tf.keras.metrics.FalsePositives(name='fp'),
        #     tf.keras.metrics.TrueNegatives(name='tn'),
        #     tf.keras.metrics.FalseNegatives(name='fn'), 
        #     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        #     tf.keras.metrics.Precision(name='precision'),
        #     tf.keras.metrics.Recall(name='recall'),
        #     tf.keras.metrics.AUC(name='auc'),
        #     tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
            ]
        class_weights = class_weights[0:n_classes]
        dice_loss = sm.losses.DiceLoss(class_weights = class_weights)
        focal = sm.losses.CategoricalFocalLoss()
        total = dice_loss + (2 * focal)
        
        self.model.compile(optimizer='adam',
              loss = total,
              metrics = my_metrics,
             )
        self.model.load_weights(model_path)

    def predictOneMask(self, img):
        x1 = cv2.resize(img, (self.size[0], self.size[1]))
        x2 = np.expand_dims(x1, axis=0)
        p1 = self.model.predict(x2, verbose=0)
        p11 = np.argmax(p1, axis = 3)
        # p111  = (np.expand_dims(p11, axis=3))
        p11 = 255*np.squeeze(p11, 0)
        return p11



if __name__ == "__main__":
    root = "../Data/"
    datasets = [
            [root+"IITD_database/",['.bmp']],
            [root+"MMU/",['.bmp']],
            [root+"Generated ds/", ['.png']]
            ]

    ob = inference(size = (128, 128, 3), n_classes=2,  BACKBONE="mine", class_weights=[0.95, 0.1,0.1,0.1], model_path = 'weights_best.hdf5')

    num = 8
    total = []
    for i, data in enumerate(datasets):
        print(i)
        res = testOnDatasets(DataSetsList = [data[0]],
            ext = data[1],
            num = num)
        if i == 0:
            total = res
        else:
            total = np.concatenate([total, res], axis = 0)

    fig,ax = plt.subplots(nrows = 1, figsize=(10,5))
    plt.imshow(total, cmap="gray")
    plt.axis('off')
    plt.savefig("model1_results_datsets_other.png", bbox_inches='tight')


    a = testOnDatasets(DataSetsList = [root +"/mineDS/"],
    ext = ['.png'],
    num = num)


    res = np.concatenate([a], axis = 0)
    fig,ax = plt.subplots(nrows = 1, figsize=(10,5))
    plt.imshow(res, cmap="gray")
    plt.axis('off')
    plt.savefig("model1_results_datsets_other_mineDS.png", bbox_inches='tight')






