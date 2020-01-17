import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Flatten
from keras.layers import Dense

train_X = np.load('trainCell.npy')
train_y = np.load('trainLabel.npy')

test_X = np.load('testCell.npy')
test_y = np.load('testLabel.npy')

eval_X = np.load('evalCell.npy')
eval_y = np.load('evalLabel.npy')
print(train_X.shape,test_X.shape,eval_X.shape)

def hot_encoder(y):
    y=(np.c_[y,1-y])
    return y

train_y=hot_encoder(train_y)
test_y=hot_encoder(test_y)
eval_y=hot_encoder(eval_y)

model=Sequential()
model.add(Dense(units=128,input_dim=16384,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=2,activation='softmax'))

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_X,train_y,epochs=100, batch_size=30)

_, accuracy = model.evaluate(eval_X, eval_y)
print('Accuracy: %.2f' % (accuracy*100))

from keras.models import model_from_json
parameters=model.to_json()
with open("parameter.json","w") as file:
    file.write(parameters)
model.save_weights("classifier.h5")
file = open('/kaggle/working/parameter.json', 'r')
model_loaded = file.read()
file.close()
classifier1 = model_from_json(model_loaded)
classifier1.load_weights("classifier.h5")

#sgd = optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
classifier1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
classifier1.evaluate(test_X,test_y)
