
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Force to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from glob import glob

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.utils import np_utils

# Configuration
batch_size = 64
nb_classes = 3
epochs = 10
img_rows, img_cols = 200, 200
input_shape = (img_rows,img_cols)
nb_pool = 2
nb_conv = 3

# Prepare input 
TRAIN_DATA = "/data/kaggle/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
TRAIN_DATA = "/data/kaggle/additional"
type_1_files += glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_2_files += glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_3_files += glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))

type_1_files = type_1_files[0:200]
type_2_files = type_2_files[0:200]
type_3_files = type_3_files[0:200]
files = type_1_files + type_2_files + type_3_files

type_1_num = len(type_1_files)
type_2_num = len(type_2_files)
type_3_num = len(type_3_files)
print "Type1 : %d Type2 : %d Type3 : %d"%( type_1_num, type_2_num, type_3_num )


# Label input data
label = np.zeros( type_1_num + type_2_num + type_3_num )
label[ type_1_num : type_1_num+type_2_num-1 ] = 1
label[ type_1_num+type_2_num : ] = 2


# Shuffle the input data
shuffle = [ i for i in range(len(files)) ] 
np.random.shuffle(shuffle)
shuffle_files = np.asarray(files)[shuffle] 
shuffle_label = label[shuffle]


# Separe train and validation data
data = np.asarray([ cv2.resize(cv2.imread(f),input_shape) for f in shuffle_files])
data = data.astype('float32')/255
data_train = data[0:int(round(len(files)*0.9))]
data_test  = data[int(round(len(files)*0.9)+1):]
label_train = shuffle_label[0:int(round(len(files)*0.9))]
label_test  = shuffle_label[int(round(len(files)*0.9))+1:]
print('data_train shape:', data_train.shape)
print(data_train.shape[0], 'train samples')
print(data_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
label_train = np_utils.to_categorical(label_train, nb_classes)
label_test  = np_utils.to_categorical(label_test, nb_classes)

inputs = Input(shape=(img_rows, img_cols, 3, ))

# Build model
M = Conv2D(16, nb_conv, nb_conv, border_mode='valid')(inputs)
M = Activation('relu')(M)
M = MaxPooling2D(pool_size=(nb_pool, nb_pool))(M)
M = Conv2D(64, nb_conv, nb_conv)(M)
M = Activation('relu')(M)
M = MaxPooling2D(pool_size=(nb_pool, nb_pool))(M)
M = Dropout(0.25)(M)
M = Conv2DTranspose(16,(nb_conv,nb_conv),strides=(2,2))(M)
M = Conv2DTranspose(3,(nb_conv,nb_conv),strides=(2,2))(M)
M = MaxPooling2D(pool_size=(nb_pool, nb_pool))(M)
maskOut = Dropout(0.25)(M)
mask = Model(inputs,maskOut)

M = Conv2D(16, nb_conv, nb_conv)(inputs)
M = Activation('relu')(M)
M = Conv2D(64, nb_conv, nb_conv)(M)
M = Activation('relu')(M)
M = MaxPooling2D(pool_size=(nb_pool, nb_pool))(M)
M = Dropout(0.25)(M)
M = Flatten()(M)
M = Dense(128)(M)
M = Activation('relu')(M)
M = Dropout(0.5)(M)
M = Dense(nb_classes)(M)
detecteroOut = Activation('softmax')(M)
dectector = Model(inputs,detectorOut)

detector.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

M = mask(inputs)
model = detector(M)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Train 
detector.fit(data_train, label_train, batch_size=batch_size, epochs=epochs, \
        verbose=1, validation_data=(data_test, label_test))
model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs, \
        verbose=1, validation_data=(data_test, label_test))

# Evaluate
score = model.evaluate(data_test, label_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


