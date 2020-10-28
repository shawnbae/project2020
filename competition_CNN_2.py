import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.activations import sigmoid


TRAIN_DIR= './NIPA_하반기 경진대회_사전검증/train'
train_datagen= ImageDataGenerator(rescale= 1./255)
train_generator= train_datagen.flow_from_directory(TRAIN_DIR,
  target_size= (256, 256),
  class_mode="categorical")
train_generator.reset()

TEST_DIR= './NIPA_하반기 경진대회_사전검증/test'
test_datagen= ImageDataGenerator(rescale= 1./255)
test_generator= test_datagen.flow_from_directory(TEST_DIR,
  target_size= (256, 256),
  class_mode="categorical")
test_generator.reset()

from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(input_shape = (256, 256, 3), # Shape of our images
                include_top = False, # Leave out the last fully connected layer
                weights = 'imagenet')   
x = Flatten()(base_model.output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = Dense(20, activation='softmax')(x)
model = Model(base_model.input, x)
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'categorical_crossentropy',metrics = ['acc'])

vgghist = model.fit(train_generator, steps_per_epoch = 1000, epochs = 10)

for layer in base_model.layers:
    layer.trainable = False


filenames = test_generator.filenames
nb_samples= len(filenames)
predict = base_model.predict_generator(test_generator,steps = nb_samples)

predicted_class_indices=np.argmax(predict ,axis=1)

