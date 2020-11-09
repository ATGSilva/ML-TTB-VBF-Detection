'''Creates a model identical to the one we found to be the best 
implementable TTB detector network. Also contains reshape functions
for changing the shapes of the histograms. The architecture already 
given was optimised for 25 by 25 image granularity.'''


import tensorflow as tf
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from tensorflow import keras
import pickle 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from scipy import mean 


#BASE_PATH="/storage/ec6821/L1TJets/MsciProjects/2020/ML-TTB-VBF-Detection/TestTrainDataCreation/"
BASE_PATH="/storage2/ec6821/P2JetsSums/MSciProjects/2020/ML-TTB-VBF-Detection/TestTrainDataCreation/"

pickle_in = open(BASE_PATH+'/test_training_x.pkl',"rb")
X = pickle.load(pickle_in)

pickle_in = open(BASE_PATH+'/test_training_y.pkl',"rb")
y = pickle.load(pickle_in)

pickle_in = open(BASE_PATH+'/test_validation_x.pkl',"rb")
X_test = pickle.load(pickle_in)

pickle_in = open(BASE_PATH+'/test_validation_y.pkl',"rb")
y_test = pickle.load(pickle_in)


def reshape_function_train(length, width):
    A = []
    for i in range(0, len(X)):
        A.append(cv2.resize(X[i][:,:,0], (length,width)))
    A = np.array(A).reshape(-1, length, width, 1)
    A = np.array(A)
    A = np.array(A/np.amax(A))
    return A 

def reshape_function_validate(length, width):
    A = []
    for i in range(0, len(X_test)):
        A.append(cv2.resize(X_test[i][:,:,0], (length, width)))
    A = np.array(A).reshape(-1, length, width, 1)
    A = np.array(A)
    A = np.array(A/np.amax(A))
    return A 


model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(9, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(60, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(35, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer='Adam',  
                  loss='binary_crossentropy',  
                  metrics=['accuracy'])#, verbosity = 1) # EJC Commented out verbosity option, doesn't work in 2020 setup
model.fit(reshape_function_train(25,25), y, epochs=3)
model.evaluate(reshape_function_validate(25, 25), y_test)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
