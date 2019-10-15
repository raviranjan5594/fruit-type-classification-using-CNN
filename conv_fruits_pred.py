#!/usr/bin/env python
# coding: utf-8

# In[84]:


#https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import os
import cv2


# In[85]:


#path to fruits directory
PATH = r"C:\Users\h\Desktop\AI_Model\DeepLearning\fruits_sample"
categorys = ["Apple", "Avocado", "Banana", "Beetroot", "Blueberry", "Carambula"]


# In[86]:


IMG_SIZE = 70
def create_training_data():
    training_data = []
    for categ in categorys:
        path = os.path.join(PATH, categ)   
        class_index = categorys.index(categ) # storing index of fruits for classification
        for img in os.listdir(path):               
            try:
                    img_array = cv2.imread(os.path.join(path, img))
                    new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_img_array, class_index])
            except Exception as e:
                pass
    return training_data


# In[87]:


trainee_data=create_training_data()
import random 
random.shuffle(trainee_data)
x=[]
y=[]
for features, label in trainee_data:
    x.append(features)
    y.append(label)
x = x/x[0][0]
print(np.shape(x))


# In[88]:


model = Sequential()
model.add( Conv2D(64, (3,3), input_shape = x.shape[1:])) # Conv layer with 70 bunit, 3x3 window size, i/p shape
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[89]:


#hidden layer 1

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[90]:


#hidden layer 2

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


# In[91]:


#hidden layer 2

model.add(Flatten())
model.add(Dense(64))


# In[92]:


#output layer
model.add(Dense(6))
model.add(Activation("softmax"))


# In[93]:


model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


# In[97]:


model.fit(x, y, batch_size=15, epochs=4, validation_split=0.30)


# In[105]:


def fruits_prediction(x):
    path_to_apple=x
    img_array_to_test = cv2.imread(path_to_apple)
    img_array_to_test = cv2.resize(img_array_to_test, (70, 70))
    plt.imshow(img_array_to_test)
    test_img_list = []
    test_img_list.append(img_array_to_test)
    test_img_list = test_img_list / test_img_list[0][0]
    #till here image pre-processing
    pre = model.predict([test_img_list])
    print(pre)
    print(np.argmax(pre))
    class_v = np.argmax(pre)
    if class_v < 6 and class_v > -1:
        print(categorys[class_v])
    else:
        print("Sorry Try somthing else!!")


# In[187]:


fruits_prediction(r'C:\Users\h\Desktop\AI_Model\DeepLearning\save_to_path.jpg')


# In[ ]:





# In[ ]:




