
import tensorflow as tf
from tensorflow.keras.layers import( Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation, MaxPool2D)

#import torch
#from torch.nn import Activation#Conv2d, BatchNormalization, ReLU, MaxPool2d, ConvTranspose2d,                         
#import torch.nn as nn

from tensorflow.keras.models import Model
import segmentation_models as sm


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 
                  3,  # filter size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
                  
    conv = BatchNormalization()(conv, training=False)
    conv = Activation("relu")(conv)

    #kernel_initializer='he_normal'
    conv = Conv2D(n_filters, 
                  3,  # filter size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization()(conv, training=False)
    conv = Activation("relu")(conv)

    if dropout_prob > 0:     
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv
    skip_connection = conv    
    return skip_connection, next_layer

def DecoderMiniBlock(prev_layer_input, skip_layer_input, strides=(2, 2), n_filters=32):
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),
                 strides=strides,
                 padding='same')(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)
    conv = Conv2D(n_filters, 
                 3,  
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3, 
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv



def build_unet(n_classes = 4, IMG_SIZE = (256, 256, 3)):
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = IMG_SIZE

    inputs =  Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),name = "Inputs")
    # define input layer
    resize_and_rescale = tf.keras.Sequential([
      #tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
      tf.keras.layers.Rescaling(1./255),
      #tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    s = inputs
    s = resize_and_rescale(s)
    
    
    s1, p1 = EncoderMiniBlock(s, n_filters=16,  dropout_prob=0, max_pooling=True)
    s2, p2 = EncoderMiniBlock(p1,     n_filters=32, dropout_prob=0, max_pooling=True)
    s3, p3 = EncoderMiniBlock(p2,     n_filters=64, dropout_prob=0, max_pooling=True)
    s4, p4 = EncoderMiniBlock(p3,     n_filters=128, dropout_prob=0, max_pooling=True)
    b1, b1 = EncoderMiniBlock(p4,     n_filters=128, dropout_prob=0, max_pooling=False)

    d1 = DecoderMiniBlock(b1, s4, n_filters=128)
    d2 = DecoderMiniBlock(d1, s3, n_filters=64)
    d3 = DecoderMiniBlock(d2, s2, n_filters=32)
    d4 = DecoderMiniBlock(d3, s1, n_filters=16)
    
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")
    #print(model.summary())
    return model


def build_unet1(n_classes = 4, IMG_SIZE = (160, 120, 3)):
    IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = IMG_SIZE
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),name = "Inputs")
    # define input layer
    resize_and_rescale = tf.keras.Sequential([
      #tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
      tf.keras.layers.Rescaling(1./255),
      #tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    s = inputs
    s = resize_and_rescale(s)
    
    
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    #c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name = "c1")(c1)
    p1 = MaxPooling2D((2, 2),name = "p1")(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    #c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name = "c2")(c2)
    p2 = MaxPooling2D((2, 2),name = "p2")(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    #c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c3")(c3)
    p3 = MaxPooling2D((2, 2),name = "p3")(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    #c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c4")(c4)
    p4 = MaxPooling2D(pool_size=(2, 2),name = "p4")(c4)
    
    # Bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "bridge")(c5)
    
    
    #------------
    #print("input: ", s.shape)
    #print(c1.shape, p1.shape)
    #print(c2.shape, p2.shape)
    #print(c3.shape, p3.shape)
    #print(c4.shape, p4.shape)
    #print(c5.shape)
    #----------
    # Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',name = "u6")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c6")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',name = "u7")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c7")(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',name = "u8")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    #c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c8")(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same',name = "u9")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    #c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c9")(c9)
    
    # define output layer
    outputs = Conv2D(n_classes, (1, 1), activation='softmax',name = "Output")(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name = "MNet-1")
    return model
    
    



#----------------------
# https://youtu.be/L5iV5BHkMzM
"""
Attention U-net:
https://arxiv.org/pdf/1804.03999.pdf
Recurrent residual Unet (R2U-Net) paper
https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
(Check fig 4.)
Note: Batch normalization should be performed over channels after a convolution, 
In the following code axis is set to 3 as our inputs are of shape 
[None, height, width, channel]. Channel is axis=3.
Original code from below link but heavily modified.
https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
"""

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K



'''
A few useful metrics and losses
'''

# jacard_coef for loss:  https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/zf_unet_224_model.py
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def dice_coef(y_true, y_pred, smooth = 1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)# -1 ultiplied as we want to minimize this value as loss function


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


## intersection over union
# def IoU(y_true, y_pred, eps=1e-6):
#     if np.max(y_true) == 0.0:
#         return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
#     intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#     union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
#     return -K.mean( (intersection + eps) / (union + eps), axis=0




def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function




##############################################################
'''
Useful blocks to build Unet
conv - BN - Activation - conv - BN - Activation - Dropout (if enabled)
'''

############################################################## Orginal UNet


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
   
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
   
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
   
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model 
    model = models.Model(inputs, conv_final, name="UNet")
    print(model.summary())
    return model





############################################################## Attention_UNet

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    
    
# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model
#-----------------------


############################################################## Attention_ResUNet

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet, with attention 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionResUNet")
    return model

def aru_conv(inputt, num_filters, drop_rate = 0, is_acti_before = False, BN = False):
    c1 = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    
    c1 = Dropout(drop_rate)(c1)
    
    c1 = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name = "c1")(c1)
    #conv = layers.Activation('relu')(conv) 
    if is_acti_before:
      p1 = MaxPooling2D((2, 2),name = "p1")(c1)
    
    s1 = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(s)
    if BN:
      s1 = layers.BatchNormalization(axis=3)(s1)
    r1 = layers.add([s1, c1])
    if not is_acti_before:
      r1 = layers.Activation('relu')(res_path)
    p1 = layers.MaxPooling2D(pool_size=(2,2))(r1)
    return c1, p1



def build_att_res_mnet(n_classes = 4, IMG_SIZE = (256, 256, 3)):
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = IMG_SIZE
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),name = "Inputs")
    # define input layer
    resize_and_rescale = tf.keras.Sequential([
      #tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
      tf.keras.layers.Rescaling(1./255),
      #tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    s = inputs
    s = resize_and_rescale(s)
    
    
    
    # Encoder
    
    c1, p1 = aru_conv(s, num_filters = 16, drop_rate = 0, is_acti_before = False, BN = False)
    c2, p2 = aru_conv(p1, num_filters = 32, drop_rate = 0, is_acti_before = False, BN = False)
    c3, p3 = aru_conv(p2, num_filters = 64, drop_rate = 0, is_acti_before = False, BN = False)
    c4, p4 = aru_conv(p3, num_filters = 128, drop_rate = 0, is_acti_before = False, BN = False)
    
    # Bridge
    c5, p5 = aru_conv(p4, num_filters = 256, drop_rate = 0.3, is_acti_before = False, BN = False)
    
    #Decoder
    
    


    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    #c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name = "c2")(c2)
    p2 = MaxPooling2D((2, 2),name = "p2")(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    #c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c3")(c3)
    p3 = MaxPooling2D((2, 2),name = "p3")(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    #c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c4")(c4)
    p4 = MaxPooling2D(pool_size=(2, 2),name = "p4")(c4)
    
    # Bridge
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "bridge")(c5)
    # Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',name = "u6")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c6")(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',name = "u7")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c7")(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',name = "u8")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    #c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c8")(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same',name = "u9")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    #c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',name = "c9")(c9)
    
    # define output layer
    outputs = Conv2D(n_classes, (1, 1), activation='softmax',name = "Output")(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name = "MNet-1")
    return model
    
    








def get_model(name = "mine", n_classes = 2, IMG_SIZE = (160, 120, 3) ):
    #print("Backbone is: ", name)
    if name == "mine":
        #return build_unet(n_classes=n_classes, IMG_SIZE = IMG_SIZE)
        return build_unet1(n_classes = n_classes, IMG_SIZE = IMG_SIZE)
    elif name == "UNET":
        return UNet(input_shape = IMG_SIZE, NUM_CLASSES = n_classes, dropout_rate = 0.0, batch_norm = True)
    elif name == "ATT_RES_UNET":
        return Attention_ResUNet(input_shape = IMG_SIZE, NUM_CLASSES = n_classes, dropout_rate = 0.0, batch_norm = True)
    elif name == "resnet34":
        return sm.Unet('resnet34', encoder_weights = 'imagenet', classes = n_classes, activation = 'softmax')
    elif name == "inceptionv3":
        return sm.Unet('inceptionv3', encoder_weights = 'imagenet', classes = n_classes, activation = 'softmax')
    elif name == "vgg16":
        return sm.Unet('vgg16', encoder_weights = 'imagenet', classes = n_classes, activation = 'softmax')
    elif name == "linknet":
        return sm.Linknet('resnet34', encoder_weights = 'imagenet', classes = n_classes, activation = 'softmax')


#IoU = TP / (TP + FP + FN)
def get_msh(a, b):
  """
  m = tf.keras.metrics.TruePositives()
  m.update_state(a, b)
  tp = m.result().numpy()
  
  m = tf.keras.metrics.FalsePositives()
  m.update_state(a, b)
  fp = m.result().numpy()
  
  m = tf.keras.metrics.FalseNegatives()
  m.update_state(a, b)
  fn = m.result().numpy()
  """  
  return 99
  
import numpy as np




"""
class MyMetrics_layer(tf.keras.metrics.Metric):

  def __init__(self, name='MyMetrics_layer', **kwargs):
    super(MyMetrics_layer, self).__init__(name=name, **kwargs)
    self.my_metric = {0:0, 1:0}#self.add_weight(name='my_metric1', initializer='zeros')
    #self.m = tf.keras.metrics.SparseCategoricalAccuracy()

  def get_rates(self, pred_labels, true_labels):
      pred_labels = pred_labels.numpy()
      true_labels = true_labels.numpy()
      
      ious = {}
      classes_id = [0, 1]
      for class_id in classes_id:
          TP = np.sum(np.logical_and(pred_labels == class_id, true_labels == class_id))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
          TN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (1-class_id)))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
          FP = np.sum(np.logical_and(pred_labels == (class_id), true_labels == (1-class_id)))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
          FN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (class_id)))
          # IoU = TP / (TP + FP + FN)
          ious[class_id] = TP / (TP + FP + FN)
      return ious
      
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.my_metric = get_rates(y_true, y_pred)
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    #self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  def reset_state(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric = {0:0, 1:0}#.assign(0.)


class MyMetrics_layer(tf.keras.metrics.Metric):

  def __init__(self, name='MyMetrics_layer', class_id = [0], **kwargs):
    super(MyMetrics_layer, self).__init__(name=name, **kwargs)
    self.my_metric = None#self.add_weight(name='my_metric1', initializer='zeros')
    #self.m = tf.keras.metrics.SparseCategoricalAccuracy()
    self.class_id = class_id

  def get_rates(self, pred_labels, true_labels, class_id):
    pred_labels = pred_labels.numpy()
    true_labels = true_labels.numpy()
    

    ious = {}
    total = 0.0

    classes_id = class_id#[0, 1]#np.array(list(set(pred_labels) | set(true_labels))).astype(int)
    for class_id in classes_id:
        TP = np.sum(np.logical_and(pred_labels == class_id, true_labels == class_id))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (1-class_id)))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(pred_labels == (class_id), true_labels == (1-class_id)))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (class_id)))
        # IoU = TP / (TP + FP + FN)
        ious[class_id] = TP / (TP + FP + FN)
        total += ious[class_id]
    ious["mean1"] = total / len(classes_id)
    m = tf.keras.metrics.MeanIoU(num_classes=len(classes_id))
    m.update_state(pred_labels, true_labels)
    ious["mean2"] = m.result().numpy()    
    #ious = torch.FloatTensor(list(ious.values()))
    return ious
      
  def update_state(self, y_true, y_pred, sample_weight=None):
    self.my_metric = self.get_rates(y_true, y_pred, self.class_id)[self.class_id[0]]
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    #self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  def reset_state(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric = None#.assign(0.)
    
"""

def get_rates(pred_labels, true_labels):
    pred_labels = pred_labels.numpy()
    true_labels = true_labels.numpy()
    
    #print(pred_labels)
    #print(true_labels)
    
    ious = {}
    total = 0.0

    classes_id = [0, 1]#np.array(list(set(pred_labels) | set(true_labels))).astype(int)
    for class_id in classes_id:
        TP = np.sum(np.logical_and(pred_labels == class_id, true_labels == class_id))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (1-class_id)))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(pred_labels == (class_id), true_labels == (1-class_id)))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (class_id)))
        # IoU = TP / (TP + FP + FN)
        ious[class_id] = TP / (TP + FP + FN)
        total += ious[class_id]
    ious["mean1"] = total / len(classes_id)
    m = tf.keras.metrics.MeanIoU(num_classes=len(classes_id))
    m.update_state(pred_labels, true_labels)
    ious["mean2"] = m.result().numpy()    
    
    #x = torch.FloatTensor(list(ious.values()))
    return ious   
    
class MyMetrics_layer(tf.keras.metrics.Metric):

  def __init__(self, name='MyMetrics_layer', wanted = 0, **kwargs):
    super(MyMetrics_layer, self).__init__(name=name, **kwargs)
    self.my_metric = None#self.add_weight(name='my_metric1', initializer='zeros')
    #self.m = tf.keras.metrics.SparseCategoricalAccuracy()
    self.wanted = wanted

      
  def update_state(self, y_true, y_pred, sample_weight = None):
    self.my_metric = get_rates(y_true, y_pred)[self.wanted]
    #_ = self.m.update_state(tf.cast(y_true, 'int32'), y_pred)
    #self.my_metric.assign(self.m.result())
    
  def result(self):
    return self.my_metric

  def reset_state(self):
    # The state of the metric will be reset at the start of each epoch.
    self.my_metric = None#.assign(0.)


tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)
import torch# import FloatTensor


def iou_class0(pred_labels, true_labels):
    pred_labels = pred_labels.numpy()
    true_labels = true_labels.numpy()
    
    ious = {}
    classes_id = [0]
    for class_id in classes_id:
        TP = np.sum(np.logical_and(pred_labels == class_id, true_labels == class_id))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (1-class_id)))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(pred_labels == (class_id), true_labels == (1-class_id)))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (class_id)))
        # IoU = TP / (TP + FP + FN)
        ious[class_id] = TP / (TP + FP + FN)
    x = torch.FloatTensor([ious[0]])
    return x#torch.FloatTensor(list(ious.values()))
    

def iou_class1(pred_labels, true_labels):
    pred_labels = pred_labels.numpy()
    true_labels = true_labels.numpy()
    
    ious = {}
    classes_id = [1]
    for class_id in classes_id:
        TP = np.sum(np.logical_and(pred_labels == class_id, true_labels == class_id))  # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (1-class_id)))  # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(pred_labels == (class_id), true_labels == (1-class_id)))  # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(pred_labels == (1-class_id), true_labels == (class_id)))
        # IoU = TP / (TP + FP + FN)
        ious[class_id] = TP / (TP + FP + FN)
    x = torch.FloatTensor([ious[1]])
    return x#torch.FloatTensor(list(ious.values()))    




def get_compiled_model(BACKBONE, n_classes, IMG_SIZE, class_weights, model_path = None):
    print("model_path: ", model_path)
    model = get_model(name = BACKBONE, n_classes = n_classes, IMG_SIZE = IMG_SIZE)
    my_metrics = [
        sm.metrics.IOUScore(threshold = 0.5),
        sm.metrics.FScore(threshold = 0.5),
        jacard_coef,
        tf.keras.metrics.MeanIoU(num_classes = n_classes),
        dice_coef,
        #tf.keras.metrics.TruePositives(name='tp'),
        #tf.keras.metrics.FalsePositives(name='fp'),
        #tf.keras.metrics.TrueNegatives(name='tn'),
        #tf.keras.metrics.FalseNegatives(name='fn'), 
    #     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        #get_rates(wanted = 0),
        #get_rates(wanted = 1),
        #get_rates(wanted = "mean1"),
        iou_class0,
        iou_class1,
        #get_rates,
        MyMetrics_layer(name='iou_pupil', wanted = 0),
        MyMetrics_layer(name='iou_bg', wanted = 1),
        MyMetrics_layer(name='iou_mean1', wanted = "mean1"),
        MyMetrics_layer(name='iou_mean2', wanted = "mean2"),
        
        ##tf.keras.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
        #tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]),
        #tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
    ]
    class_weights = class_weights[0:n_classes]
    dice_loss = sm.losses.DiceLoss(class_weights = class_weights)
    focal = sm.losses.CategoricalFocalLoss()
    total = dice_loss + (2 * focal)
    
    model.compile(optimizer='adam',
          loss = total,
          metrics = my_metrics,
         )
    if model_path is not None:
        model.load_weights(model_path)
    #print(model.summary())

    return model

def preprocessing_model_data(BACKBONE, X_train1, X_val1, X_test1):
    if BACKBONE not in ["mine", "linknet", "ATT_RES_UNET", "UNET"]:
        preprocess_input1 = sm.get_preprocessing(BACKBONE)
        # preprocess input
        X_train1 = preprocess_input1(X_train1)
        X_test1 = preprocess_input1(X_test1)
        X_val1 = preprocess_input1(X_val1)    
    return X_train1, X_val1, X_test1
