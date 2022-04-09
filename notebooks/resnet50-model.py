#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


IMG_DIR = pathlib.Path('C:\\Users\\USER-PC\\Documents\\EE4208\\resized_300\\resized_300x300')


# In[3]:


import glob
U1 = glob.glob('resized_300/resized_300x300/User1/*.*')
U2 = glob.glob('resized_300/resized_300x300/User2/*.*')
U3 = glob.glob('resized_300/resized_300x300/User3/*.*')
U4 = glob.glob('resized_300/resized_300x300/User4/*.*')

U5 = glob.glob('resized_300/resized_300x300/User5/*.*')
U6 = glob.glob('resized_300/resized_300x300/User6/*.*')
U7 = glob.glob('resized_300/resized_300x300/User7/*.*')
U8 = glob.glob('resized_300/resized_300x300/User8/*.*')

U9 = glob.glob('resized_300/resized_300x300/User9/*.*')
U10 = glob.glob('resized_300/resized_300x300/User10/*.*')
U11 = glob.glob('resized_300/resized_300x300/User11/*.*')
U12 = glob.glob('resized_300/resized_300x300/User12/*.*')

U13 = glob.glob('resized_300/resized_300x300/User13/*.*')
U14 = glob.glob('resized_300/resized_300x300/User14/*.*')
U15 = glob.glob('resized_300/resized_300x300/User15/*.*')
U16 = glob.glob('resized_300/resized_300x300/User16/*.*')

U17 = glob.glob('resized_300/resized_300x300/User17/*.*')
U18 = glob.glob('resized_300/resized_300x300/User18/*.*')
U19 = glob.glob('resized_300/resized_300x300/User19/*.*')
U20 = glob.glob('resized_300/resized_300x300/User20/*.*')

U21 = glob.glob('resized_300/resized_300x300/User21/*.*')
U22 = glob.glob('resized_300/resized_300x300/User22/*.*')
U23 = glob.glob('resized_300/resized_300x300/User23/*.*')
U24 = glob.glob('resized_300/resized_300x300/User24/*.*')

data = []
labels = []

for i in U1:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(0)
for i in U2:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(1)
for i in U3:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(2)
for i in U4:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(3)
for i in U5:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(4)
for i in U6:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(5)
for i in U7:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(6)
for i in U8:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(7)
for i in U9:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(8)
for i in U10:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(9)
for i in U11:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(10)
for i in U12:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(11)
for i in U13:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(12)
for i in U14:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(13)
for i in U15:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(14)
for i in U16:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(15)
for i in U17:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(16)
for i in U18:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(17)
for i in U19:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(18)
for i in U20:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(19)
for i in U21:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(20)
for i in U22:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(21)
for i in U23:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(22)
for i in U24:   
    image=tf.keras.preprocessing.image.load_img(i)
    image=np.array(image)
    data.append(image)
    labels.append(23)


data = np.array(data)
labels = np.array(labels)

from sklearn.model_selection import train_test_split
X_train, X_test, ytrain, ytest = train_test_split(data, labels, test_size=0.2,
                                                random_state=42)


# In[4]:


train_datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,  
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)

train_datagen.fit(X_train)


# In[5]:


from tensorflow.keras.models import Model

resnet = ResNet50(include_top=False, weights='imagenet',pooling='max')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)


# In[6]:


resnet.summary()


# In[7]:


print(labels)


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

num_classes = 24

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# In[9]:


adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                              patience=8,
                                              min_delta=0.001,
                                              verbose=1,
                                              mode='auto'
                                             )

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.2,
                                   patience=4,
                                   verbose=1,
                                   min_delta=5*1e-3,min_lr = 5*1e-7,
                                   )

callbacks = [early_stop, reduce_lr]


# In[10]:


import tensorflow_addons as tfa

model.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
#tfa.metrics.F1Score(num_classes=num_classes)


# In[11]:


batch_size = 8

history = model.fit(
    train_datagen.flow(X_train,ytrain,batch_size),
    validation_data  = (X_test,ytest),
    validation_steps = len(X_test)//batch_size,
    steps_per_epoch  = len(X_train)//batch_size,
    epochs = 25, 
    verbose=1,
    callbacks=[early_stop,reduce_lr]
)


# In[13]:


epochs=25

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(20, 11))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("accuracy = %.4f, val_accuracy = %.4f" %(max(history.history['accuracy']),max(history.history['val_accuracy'])))
print("loss = %.4f, val_loss = %.4f" %(min(history.history['loss']),min(history.history['val_loss'])))


# 
