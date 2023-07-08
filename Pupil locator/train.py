
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel('INFO')


import tensorflow, tensorflow_addons, segmentation_models 
#print(tensorflow.__version__, tensorflow_addons.__version__, segmentation_models.__version__)
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
import parsing_file
# import inference


#------------------------------user libraries
from utilis import utilitis, plot_img, ClearToBlack, read_color_map, load_data_all, prepare_data, encode_lables
# import imageProcessing
from ML_DL_utilis import MLDL_utilitis
import models as mnet
from models import jacard_coef, dice_coef, jacard_coef_loss, dice_coef_loss, jacard_coef_loss, preprocessing_model_data, get_compiled_model

mldl_uts = MLDL_utilitis()
uts = utilitis()


"""
def predict(model, X_test, y_test_cat, y_test , num = 1, show =  False, name = ""):
    inps = []
    outs = []
    if num == -1:
        inps = X_test
        outs = y_test
    else:
        #print(len(X_test), num)
        num = min(len(X_test), num)
        idx = np.argsort(np.random.randint(0, len(X_test), num))
        for i in idx:
            inps.append(X_test[i])
            outs.append(y_test[i])
    

    inps = np.array(inps)
    outs = np.array(outs)
    prediction = model.predict(inps)
    predicted_imgages1 = np.argmax(prediction, axis = 3)#[0,:,:]
    predicted_imgages  = (np.expand_dims(predicted_imgages1, axis=3))
    print(inps.shape, outs.shape, predicted_imgages.shape)
    
    if num != -1:
        s = uts.display(display_list  = [inps, outs, predicted_imgages],
                        title = ['Input Image', 'True Mask', 'Predicted'],
                        num = len(predicted_imgages),
                        size =(7, num*4/2),
                        show = True
                       );
        if name != "":
            s.savefig(f"{current_model}/{name}")
#         s = uts.display(display_list  = [inps, outs, predicted_imgages],
#                         title = ['Input Image', 'True Mask', 'Predicted'],
#                         idx = [1, 2],
#                         num = num,
#                         size =(10, 10),
#                         show = True
#                        )
#         s.savefig(f"{current_model}/{model1_results_output}")


def test(z):
    return  z.all(axis=1)
"""

    



def readParametersFromCmd():
    global EPOCHS, LEARNING_RATE, EarlyStopping, wanted, t, BACKBONE, BATCH_SIZE, outputFile, class_weights, batch_size, DATASET, metric_thr, model_weights

    parser = parsing_file.create_parser()
    args = parser.parse_args()
    EPOCHS = args.epochs 
    LEARNING_RATE = args.lr
    EarlyStopping  = args.es
    wanted = args.wanted.strip("''")
    t = args.file2read
    BACKBONE = args.backbon.strip("''")

    tmp = args.weights.strip("''").strip("[]")
    class_weights = list(map(float, tmp.split(',')))
    outputFile = "" + args.output.strip("''")
    DATASET = args.DS_NAME.strip("''")
    model_weights = args.model_weights.strip("''")

    batch_size = args.batch_size
    metric_thr = args.metric_thr
    
    #print(EPOCHS, LEARNING_RATE, EarlyStopping, wanted, t, BACKBONE, outputFile, class_weights, batch_size, DATASET, metric_thr, model_weights)

    

print("------------------------------------------------------------------------Initilazation")
readParametersFromCmd()


#SIZE = (160, 120, 3)

from utilis import load_json
SIZE = load_json("config.json")['input_size']
SIZE_X, SIZE_Y= SIZE[0], SIZE[1]
BATCH_SIZE_EVALUATE = load_json("config.json")['BS_eval']





batch_size_val, batch_size_test = 1, 1
testValRatio, testRatio = 0.30, 0.30
#BATCH_SIZE = 1
epoch_interval = 1
STEP_SIZE  =2
current_model = datetime.datetime.now().strftime("%d_%m_%Y:%H_%M_%S")
current_model = f"../Results/model1/model_{BACKBONE}_{current_model}_{outputFile}"
pathlib.Path(f'{current_model}/').mkdir(parents=True, exist_ok=True)#metrics
mldl_uts.setDir(d = f"{current_model}/")
# mldl_uts.setDir(d = f"{current_model}/metrics/")

cp_dir = f"{current_model}/checkpoints/"
cp_name = f"{current_model}/model1_results_segs_best.hdf5"

model_results_plot_architecture = "model1_results_plot_architecture.png"
model_results_saved = "model1_results_saved.h5"
model_results_output = "model1_results_output.png"
model_results_architecture = "model1_results_architecture.png"
MODEL_PATH = current_model + '/model1_results_segs_best.hdf5'



imagedata, maskdata = load_data_all(DATASET, num_to_read = t, SIZE = SIZE)
if DATASET == "NN_human_mouse_eyes":
  DATASET = '../Data/NN_human_mouse_eyes';
elif DATASET == "MOBIUS":
  DATASET = '../Data/Eye dataset/pairs';
elif DATASET == "s-openeds":
  DATASET = r"../Data/s-openeds";
elif DATASET == "ClinicAnnotated_DA":
  DATASET = r"../Data/ClinicAnnotated_DA"

s = np.random.randint(0, len(maskdata))

plot_img(maskdata[s])
plt.savefig("check_before_clearning.png")

print("------------------------------------------------------------------------read colormap:")
dsInfo, labels_color, unwanted, wanted_classes = read_color_map(file = DATASET + "/classes.json", maskdata = maskdata,  wanted = wanted)
n_classes = len(labels_color)
bg_color = np.array(dsInfo['maskColor']['Background'])


#print(wanted, unwanted, "__________________________________________")
for i, mask in enumerate(maskdata):
    for uw in unwanted:
        maskdata[i] = ClearToBlack(mask, list(uw)[1])



# s = np.random.randint(0, len(maskdata))
plot_img(maskdata[s])
plt.savefig(f"check_after_clearning_{wanted}.png")
maskdata = maskdata.copy()


X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(imagedata, maskdata, testValRatio, testRatio)

y_train_cat, y_val_cat, y_test_cat, y_cat, d = encode_lables(y_train, y_val, y_test, maskdata, labels_color)



#class_weights = class_weight.compute_class_weight('balanced', np.array(len(labels_color)), y_cat)

#class_weights1 = class_weight.compute_class_weight('balanced',
#                                                 range(len(labels_color)),
#                                                 y_train)
                                                 
class_weights = class_weights[0:n_classes]


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = SIZE#X_train.shape[1], X_train.shape[2], X_train.shape[3]
#print("n_classes: ", n_classes, "\t\t", wanted_classes, "\n\n\n")

#print("Img shape: " , (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#labels_color = list(np.array(labels_color)[:,1])

# In[12]:
print(imagedata.shape, maskdata.shape)
print("min, max", np.min(imagedata[0]), np.max(imagedata[0]))
print(f"\nimage input size: {imagedata[0].shape}\nDataset:\tImages\tMasks\n\t\t{len(imagedata)}\t{len(maskdata)}\t")


print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
print("wanted_classes", wanted_classes)
print("labels_color u: ", labels_color)
print("unwanted: ",unwanted)
print("Class weights are...:", class_weights)
print("labels_color:  ", labels_color)
labels_color = list(zip(*labels_color))[1]
print("labels_color- just:  ", labels_color)
print("dictionary:  ", d)





def scheduler1(epoch, lr):
    return lr if epoch < 10 else lr * tf.math.exp(-0.1)

def printLog(epoch, logs):
    
    if (epoch == 0):
        ss = f""
        for k in logs.keys():
            ss = ss + '{0}\t    '.format(k)
        print("Epochs\t",ss)
        
        
    if (epoch % epoch_interval) == 0:
        res = {key : round(logs[key], 3) for key in logs}
        s = f""    
        for  k in res.keys():
            s = s + '{0:04f}\t'.format(res[k])
        print(epoch+1, '\t', s)

class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (0 if epoch % self.epoch_interval != 0 else self.default_verbose)
        super().on_epoch_begin(epoch, *args, **kwargs)
 
 
callbacks =  [
#     tf.keras.callbacks.LearningRateScheduler(scheduler),
    tfa.callbacks.TQDMProgressBar(leave_epoch_progress = True, leave_overall_progress = True, show_epoch_progress = False,show_overall_progress = True),
    #tf.keras.callbacks.LambdaCallback(on_epoch_end = printLog),
    ModelCheckpoint(cp_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True),
    
#     tf.keras.callbacks.ModelCheckpoint(cp_dir+cp_name, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode="auto"),
    #tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)
# tf.keras.callbacks.TensorBoard(log_dir=f"{current_model}/logs/fit/", histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1),
] 


print("EarlyStopping", EarlyStopping)
if EarlyStopping == 1:
  print("EarlyStopping")
  callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="min", restore_best_weights=False))


if model_weights != '':
    print("init model\n\n")
    model = get_compiled_model(BACKBONE = BACKBONE, n_classes = n_classes, IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), class_weights = class_weights,  model_path = model_weights)
else:
    model = get_compiled_model(BACKBONE = BACKBONE, n_classes = n_classes, IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), class_weights = class_weights,  model_path = None)

print(model.summary())
model_tmp1 = model
X_train, X_val, X_test = preprocessing_model_data(BACKBONE, X_train, X_val, X_test)
_ = mldl_uts.saveModelArchitecture(model = model, fn = f"{model_results_architecture}", save = True);
start = datetime.datetime.now() 
history = model.fit(x = X_train,
                    y = y_train_cat ,
                    epochs = EPOCHS,
                    validation_data = (X_val, y_val_cat),
                    callbacks = callbacks,
                    shuffle = True,
                    verbose = 0,
                    batch_size= batch_size
                   )
                   
ellapsed_time = datetime.datetime.now() - start


mldl_uts.plotHistory(history, n = [1, 2], size = (10,10), show = False, prefix = "model1_results_")
tf.keras.utils.plot_model(model_tmp1,
                          to_file = f"{current_model}/{model_results_plot_architecture}",
                          show_shapes=True,
                          show_dtype=False,#do not
                          show_layer_names=False,#do not
                          rankdir='TB',
                          expand_nested=True,
                          dpi=96,
                          layer_range=None,
                         );

# import visualkeras
# #font = ImageFont.truetype("arial.ttf", 32)  # using comic sans is strictly prohibited!
# visualkeras.layered_view(model,legend=True, draw_volume=True, to_file='output.png')
#---------------save and load model
model.save(f"{current_model}/{model_results_saved}")
# model.save(f"{current_model}/model_way3")
# tf.saved_model.save(model, f"{current_model}/model_way2");
# del model
# model_ev = tf.saved_model.load(f"{current_model}/model_way2/")


# In[ ]:


#----------------- load weights
# print(cp_dir)
# print(f"The checkpoints: in :\t{cp_dir}\n")
# os.listdir(cp_dir)
# del model
# model_ev = get_model()
# model_ev.compile(optimizer='adam', loss=total, metrics=["accuracy", jacard_coef])
# model_ev.load_weights(f"{cp_dir}/{cp_name}")
# # model.load_weights(tf.train.latest_checkpoint(cp_dir))


# In[ ]:


# model = load_model(
#     f"{current_model}{model}/model.pth",
#     custom_objects={'jaccard_index': jaccard_index}
# )


# ## Evaluate

# In[57]:


# Evaluate the model on the test data using `evaluate`
print("------------------------------------------------------------------------Evaluate on test data")





evalResults = model.evaluate(X_test, y_test_cat, batch_size=BATCH_SIZE_EVALUATE, verbose = 0, return_dict=True)
trainResults1 = history.history#model.get_metrics_result()
#print("Training results: ", trainResults1)
#print("Evaluation results: ", evalResults)

print(f"Training\t\tEvaluation\n")



metrics_list = ['mean_io_u', 'f1-score', 'loss', 'precision', 'recall', 'auc', 'prc', 'dice_coef', 'iou_score', 'jacard_coef']

trainResults = {}
for k, v in trainResults1.items():
  if k == "loss":
    val_t = min(v)
  else:
    val_t = max(v)
  trainResults[k] = val_t


#trainResults = pd.DataFrame.from_dict(trainResults)
#evalResults = pd.DataFrame.from_dict(evalResults)

data = []
for m in metrics_list:
    if m in trainResults.keys():
        data.append([m, np.round(trainResults[m], 4), np.round(evalResults[m],4)])
        
train_eval_res = pd.DataFrame(data, columns = ["metric", "Training", "Evaluation"])
print(train_eval_res)
train_eval_res.to_csv(f"{current_model}/train_eval_res.csv", index = False)


print("execution time is: ", ellapsed_time)

print("-----------------------------------------------------------------------Test samples")

y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis = 3)
#print(y_pred_argmax.shape)


def calc_MIoU_IIoU(y_test, y_pred, n_classes):
    IoU_value = MeanIoU(num_classes = n_classes)
    IoU_value.update_state(y_test, y_pred)
    values = np.array(IoU_value.get_weights()).reshape(n_classes, n_classes)
#     values =  np.random.randint(1, 5, size=(n_classes, n_classes))
#     values = np.array(
#     [[1,1,3,1,2],
#     [4,2,4,4,2],
#     [3,4,3,3,4],
#     [1,1,1,1,4],
#     [3,3,4,4,1]])
#     print(values)
    res = [IoU_value.result().numpy()]
    for i in range(n_classes):
        t1 = values[i, i:].sum() +  values[i+1:, i].sum()
        t = values[i,i] / t1
        res.append(t)
    return res
    
    
res = calc_MIoU_IIoU(y_test_cat[:, :, :, 0], y_pred_argmax, n_classes)
print("IOU: ", res)

#predict(model, X_test, y_test_cat, y_test , num = 4, name = model_results_output)
#predict(model, X_test, y_test_cat, y_test , num = 3, name = "add.png")


from eval import predictOnSome, inference
from evaluate_all import myTasks


ob = inference(size = SIZE, n_classes = n_classes,  BACKBONE = "mine", class_weights = class_weights, model_path = MODEL_PATH)

res1 = predictOnSome(X_test, y_test, num = 8, ob1 = ob)  
#res2 = predictOnSome(X_test, y_test, num = 4, ob1 = ob)  
#cv2.imwrite(f"{current_model}/{model_results_output}", np.concatenate([res1, res2], axis=0))
cv2.imwrite(f"{current_model}/{model_results_output}", res1)


print("\n\n\n\n")



out = f'{current_model}/evaluation/'
pathlib.Path(out).mkdir(parents=True, exist_ok=True)

myTasks(out, MODEL_PATH, 2, wanted)
# ob = inference.inference(size = SIZE, n_classes=n_classes,  BACKBONE="mine", class_weights=class_weights, model_path = current_model + '/model1_results_segs_best.hdf5')
# inference.testOnAnother(current_model + "/", ob = ob)

a1  = cv2.imread(out + "res_ClinicAnnotated_DA.png")
b1  = cv2.imread(out + "res_NN_human_mouse_eyes.png")
c1  = cv2.imread(out + "res_MOBIUS.png")

a2  = cv2.cvtColor(cv2.imread(out + "eval_IITD_database.png"), cv2.COLOR_BGR2RGB)
b2  = cv2.cvtColor(cv2.imread(out + "eval_LPW.png"), cv2.COLOR_BGR2RGB)
c2  = cv2.cvtColor(cv2.imread(out + "eval_MMU-Iris-Database.png"), cv2.COLOR_BGR2RGB)


res1 = np.concatenate([a1, b1, c1])
res2 = np.concatenate([a2, b2,c2])

cv2.imwrite(f"{current_model}/model_all_test1.png", res1)
cv2.imwrite(f"{current_model}/model_all_test2.png", res2)

os.system(f"python3 eval.py --IMG_PATH ../toBeTested/d11.png ../toBeTested/d12.png ../toBeTested/d13.png ../toBeTested/d14.png  --OUTPUT_PATH {out}total_red_border1.png --MODEL_PATH {MODEL_PATH}")
os.system(f"python3 eval.py --IMG_PATH  ../toBeTested/d21.bmp ../toBeTested/d22.bmp ../toBeTested/d31.png ../toBeTested/d32.png ../toBeTested/d41.tif ../toBeTested/d42.tif --OUTPUT_PATH {out}total_red_border2.png --MODEL_PATH {MODEL_PATH}")



df = pd.read_csv(out + "evaluation_results_all.csv")
datasets = df["dataset_name"]
df1 =  df.T
cols_dict = {i:v for i, v in enumerate(datasets)}
df1 = df1.rename(cols_dict, axis='columns')
df1 = df1.drop('dataset_name')
df1.to_csv(out + 'evaluation_results_all.csv')



"""    
df = pd.DataFrame([classes_IoU])
col_names = ["mean"]
for c in range(n_classes):
    col_names.append(f"class {c}")
df.columns = col_names
df
"""


# In[59]:


#uts.dataframeAsImage(d = [classes_IoU], path = f"{current_model}/IoU_testing.png", rowNames = [""], colsNames = col_names,  save = True);




