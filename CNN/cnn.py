from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

def hot_encoder(y):
    y=(np.c_[y,1-y])
    return y
test_X = np.load('/kaggle/input/Test/Test/cells.npy')
test_y = np.load('/kaggle/input/Test/Test/labels.npy')

train_X = np.load('/kaggle/input/Train/Train/cells.npy')
train_y = np.load('/kaggle/input/Train/Train/labels.npy')

eval_X = np.load('/kaggle/input/Eval/Eval/cells.npy')
eval_y = np.load('/kaggle/input/Eval/Eval/labels.npy')
test_y=hot_encoder(test_y)
train_y=hot_encoder(train_y)
eval_y=hot_encoder(eval_y)
tf.compat.v1.reset_default_graph()
classifier=Sequential()
classifier.add(Conv2D(32,(3,3),padding='same',activation="relu",input_shape=(128,128,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(64,(4,4),padding='same',activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(128,(3,3),padding='same',activation="relu"))
classifier.add(MaxPooling2D(pool_size=(3,3)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dense(output_dim=32,activation='relu'))
classifier.add(Dense(output_dim=2,activation='softmax'))
opt = Adam(lr=1e-4, decay=1e-4 / 13)
classifier.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
datagen=ImageDataGenerator(horizontal_flip=True)
classifier.fit_generator(datagen.flow(train_X,train_y,batch_size=32),validation_data=(eval_X,eval_y),steps_per_epoch=len(train_X)//32,epochs=13)
from keras.models import model_from_json
parameters=classifier.to_json()
with open("parameter.json","w") as file:
    file.write(parameters)
classifier.save_weights("classifier.h5")
file = open('/kaggle/working/parameter.json', 'r')
model_loaded = file.read()
file.close()
classifier1 = model_from_json(model_loaded)
classifier1.load_weights("classifier.h5")
classifier1.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
classifier1.evaluate(test_X,test_y,verbose=0)

    
