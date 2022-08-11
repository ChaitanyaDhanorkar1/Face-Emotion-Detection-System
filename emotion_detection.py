# Importing Tensorflow Library
import tensorflow as tf

# For splliting the dataset into training and testing sets.
from sklearn.model_selection import train_test_split

from tensorflow import keras
import pandas as pd
import numpy as np
# For image visualisation
import sklearn 

import os
import matplotlib
from matplotlib import pyplot as plt 
import PIL
import PIL.Image


path='./fer2013.csv'
fer2013 = pd.read_csv(path)

fer2013.shape

fer2013.head()

print(fer2013.emotion,fer2013.Usage)

print(fer2013.pixels)

print(fer2013.pixels[0])

print(fer2013.Usage.unique())
print(fer2013.emotion.unique())

# The labels for indices is given in the descripiton of FER2013 dataset.
# https://paperswithcode.com/dataset/fer2013
label_names = {0:'Angry',1:'Disgust',2:'Fear',3:'Happiness',4:'Sadness',5:'Surprise',6:'Neutral'}

fer2013.pixels.loc[0]

fer2013.pixels.loc[0].split(' ')

np.array(fer2013.pixels.loc[0].split(' '))

# Converting the first image to numpy array.
np.array(fer2013.pixels.loc[0].split(' ')).reshape(48,48)

# plt.imshow(np.array(fer2013.pixels.loc[0].split(' ')).reshape(48,48)  --- Float Error
plt.imshow(np.array(fer2013.pixels.loc[0].split(' ')).reshape(48,48).astype(float))

plt.figure(figsize=(12,12))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    curr_img = np.array(fer2013.pixels.loc[i].split(' ')).reshape(48,48).astype(float)
    plt.imshow(curr_img)
    plt.xlabel(label_names[fer2013.emotion[i]])
plt.show()

images=fer2013.pixels.apply(lambda x:np.array(x.split(' ')).reshape(48,48,1).astype('float32'))

images = np.stack(images,axis=0)

print(images.shape)

type(images)

labels=fer2013.emotion.values
print(labels)

train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.1)

print(train_images.shape,test_images.shape,train_labels.shape,test_labels.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),
   
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    

    tf.keras.layers.Flatten(input_shape=(48, 48)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(7,activation='softmax')
])

model.fit(train_images, train_labels, epochs=15)

fer_json = model.to_json()  
with open("fer.json", "w") as json_file:  
    json_file.write(fer_json)  
model.save_weights("fer.h5")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


print(predictions[0])
print(np.argmax(predictions[0]))
print(label_names[np.argmax(predictions[0])])

