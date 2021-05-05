from sklearn.model_selection import train_test_split

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random

# 資料預處理
PATH = '/Users/bill/Desktop/shopee-product-detection-student'

"""train資料處理"""
train_dir = os.path.join(PATH, 'train')
train_dataset = list()
for label in range(42):
    print(label)
    if 0 <= label <= 9:
        label = '0' + str(label)
    else:
        label = str(label)
    temp_path = os.path.join(train_dir, label)
    for image in os.listdir(temp_path):
        data = cv2.imread(os.path.join(temp_path, str(image)), cv2.IMREAD_COLOR)
        data = cv2.resize(data, (28, 28)).astype('float32')/255
        train_dataset.append([data, label])

random.shuffle(train_dataset)

train_data = []
train_category = []
construct = False
for t_data, t_label in train_dataset:
    train_data.append(t_data)
    train_category.append(t_label)
train_data = np.array(train_data)

# 對於 Label 做 One-Hot Encoding 處理
train_category = to_categorical(train_category)
# 分組（train,val）
train_data, val_data, train_category, val_category = train_test_split(train_data, train_category, test_size=0.25, random_state=13)

print(train_data.shape)
print(train_category)

# Data augmenting
train_data_gen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             data_format='channels_last')


train_data_gen.fit(train_data)

LR_function = ReduceLROnPlateau(monitor='val_acc',
                             patience=3,
                                # 3 epochs 內acc沒下降就要調整LR
                             verbose=1,
                             factor=0.5,
                                # LR降為0.5
                             min_lr=0.00001
                                # 最小 LR 到0.00001就不再下降
                             )
# Create the model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(42, activation='softmax'))

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_data_gen.flow(train_data, train_category, batch_size=64),
                              steps_per_epoch=train_data.shape[0]/64, epochs=100, validation_data=(val_data, val_category),
                              validation_steps=val_data.shape[0]/64, callbacks=[LR_function])

probability_model = models.Sequential([model, layers.Softmax()])

"""收集test圖片"""
test_dataset = list()
test_names = list()
test_dir = os.path.join(os.path.join(os.path.join(PATH, 'test'), 'test'), 'test')
for image in os.listdir(test_dir):
    data = cv2.imread(os.path.join(test_dir, str(image)), cv2.IMREAD_COLOR)
    data = cv2.resize(data, (28, 28)).astype('float32') / 255
    test_dataset.append(data)
    test_names.append(str(image))
    print(image)
test_dataset = np.array(test_dataset)
print(test_dataset.shape)

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'category'])
    predictions = probability_model.predict(test_dataset)
    for num in range(len(test_names)):
        writer.writerow([test_names[num], np.argmax(predictions[num])])

print('Successfully Predict!')


# 抓出test要的圖片（shoppee給test12193筆只要12187筆）
test = []
with open('test.csv', 'r', newline='') as csvfile_test:
    rows = csv.reader(csvfile_test)
    a = 0
    for i in rows:
        if a > 0:
            test.append(i)
        a += 1
print(test)
output = []
with open('output.csv', 'r', newline='') as csvfile_output:
    rows = csv.reader(csvfile_output)
    a = 0
    for i in rows:
        if a > 0:
            output.append(i)
        a += 1
with open('final.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'category'])
    for i in test:
        for j in output:
            if i[0] == j[0]:
                writer.writerow([i[0], j[1]])
print('Done!!!')
