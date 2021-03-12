#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports for array-handling and plotting
import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# for testing on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.utils import np_utils


# In[2]:



(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


w = 26
h = 26
train = []
for image in X_train:
    #read images path
    img = image
    re = cv2.resize(img, (w,h))
    train.append(re)

test = []
for image in X_test:
    #read images path
    img = image
    re = cv2.resize(img, (w,h))
    test.append(re)


# In[4]:


fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig.show()


# In[5]:


train=np.array(train)
test=np.array(test)


# In[6]:


X_train = train.reshape(60000, w*h)
X_test = test.reshape(10000, w*h)

X_train=X_train/255
X_test=X_test/255


# In[7]:


n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train,n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# In[8]:


# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(384, input_shape=(w*h,)))
model.add(Activation('relu'))

model.add(Dense(394))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[9]:


model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')


# In[10]:


model.summary()


# In[11]:


model.fit(X_train, Y_train, batch_size=686, epochs=30,validation_data=(X_test, Y_test))


# In[12]:


model.evaluate(X_test,Y_test)[1]


# In[20]:


pred=model.predict(X_test)
for i in range(0,len(pred)):
    for j in range(0,len(pred[0])):
        if(pred[i][j]>=0.5):
            pred[i][j]=int(1)
        else:
            pred[i][j]=int(0)


# In[22]:


pred=np.argmax(pred,1)


# In[24]:


ytest=np.argmax(Y_test,1)


# In[26]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(ytest,pred)


# In[29]:


import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
cm_df = pd.DataFrame(cm,
                     index = np.unique(ytest), 
                     columns = np.unique(ytest))

plt.figure(figsize=(6,5))
sns.heatmap(cm_df,cmap="Blues", annot=True)
plt.title(' NN \nAccuracy:{0:.3f}'.format(accuracy_score(ytest, pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:




