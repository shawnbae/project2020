import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import sigmoid

os.chdir('C:/Users/soohan/project2020')
train_file_list= os.listdir('./NIPA_하반기 경진대회_사전검증/train')
test_file_list= os.listdir('./NIPA_하반기 경진대회_사전검증/test')

train_path= os.getcwd() + '\\NIPA_하반기 경진대회_사전검증\\train'
test_path= os.getcwd() + '\\NIPA_하반기 경진대회_사전검증\\test'

# data list 만들기
train_data= []
for file in train_file_list:
  image= Image.open(os.path.join(train_path, file))
  train_data.append(np.array(image))  

test_data= []
for file in test_file_list:
  image= Image.open(os.path.join(test_path, file))
  test_data.append(np.array(image))  

trainX= np.array(train_data)
testX= np.array(test_data)

with open(os.getcwd() + '\\NIPA_하반기 경진대회_사전검증\\train.tsv') as f:
  train_label= f.read()
train_label= train_label.split('\n')[:-1]
trainY= []
for label in train_label:
  trainY.append("".join(label.split('\t')[1:]))
trainY= pd.Categorical(trainY).rename_categories(np.arange(20))
trainY= to_categorical(trainY, num_classes=20)


# pretrained model 사용하기
pretrained_model= tf.keras.applications.VGG19(include_top=True, weights=None,\
                                                 classes= 20, input_shape=(256,256,3))
pretrained_model.compile(optimizer= Adam(lr= 0.005), loss= 'categorical_crossentropy')
pretrained_model.fit(trainX,trainY, epochs= 10)


pretrained_model2= tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None,\
                                                 classes= 20, input_shape=(256,256,3))
pretrained_model2.compile(optimizer= Adam(lr= 0.005), loss= 'categorical_crossentropy')
pretrained_model2.fit(trainX,trainY, epochs= 10)

pred= pretrained_model.predict(trainX)




