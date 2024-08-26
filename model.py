import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import time
import itertools
import random


import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
import tensorflow_io as tfio
from tensorflow.keras.applications import resnet
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from google.colab.patches import cv2_imshow
import matplotlib.image as mpimg
from tensorflow.keras import applications
from tensorflow.keras import metrics


from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
import warnings
import matplotlib.image as mpimg
from functools import reduce
warnings.filterwarnings('ignore')



input_dim = target_shape =  (180,180,3)
base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=input_dim, include_top=False
)

base_cnn.trainable=False

glob_pool = layers.GlobalAveragePooling2D()(base_cnn.output)
dense1 = layers.Dense(128)(glob_pool)
output = layers.Dense(128,activation='sigmoid')(dense1)

embedding_new = Model(base_cnn.input, output, name="Embedding")

anchor_in = Input(name='anchor',shape=target_shape)
pos_in = Input(name='positive',shape=target_shape)
neg_in = Input(name='negative',shape=target_shape)

# Extract embeddings using VGG19
anchor_out = embedding_new(anchor_in)
pos_out = embedding_new(pos_in)
neg_out = embedding_new(neg_in)
# Concatenate the embeddings

model_lossless_triplet_loss = Model(inputs=[anchor_in, pos_in, neg_in], outputs=[anchor_out,pos_out,neg_out])