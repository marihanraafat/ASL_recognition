#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Import the required libraries.
import os
import cv2
#import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[2]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('points_test') 

# Actions that we try to detect
actions = np.array(['my','father','help','me','now'])

# Thirty videos worth of data
no_sequences = 200

# Videos are going to be 30 frames in length
sequence_length = 21


# In[3]:


label_map = {label:num for num, label in enumerate(actions)}


# In[4]:


label_map


# In[5]:


sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[6]:


X = np.array(sequences)


# In[7]:


y = to_categorical(labels).astype(int)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[ ]:


from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.losses import mean_squared_error
from keras.regularizers import l2
# Define the input shape
input_shape = (X_train.shape[1], X_train.shape[2])  # (21, 1662)

# Define the number of output classes
num_classes = y_train.shape[1]  # 5

# Define the model architecture
model = Sequential()
model.add(GRU(256, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(GRU(256, kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(num_classes, activation='softmax'))

# Define the optimizer
optimizer = Adam(lr=0.0001)

# Define the learning rate schedule
def lr_schedule(epoch):
    if epoch < 10:
        return 0.0001
    else:
        return 0.00001

# Compile the model
model.compile(loss=mean_squared_error, optimizer=optimizer, metrics=['accuracy'])

# Define the callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model on larger batches
batch_size = 256
steps_per_epoch = int(len(X_train) / batch_size)
validation_steps = int(len(X_test) / batch_size)

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=batch_size, steps_per_epoch=steps_per_epoch, validation_data=(X_test, y_test), validation_steps=validation_steps, callbacks=[early_stopping, model_checkpoint, lr_scheduler])


# # predictions and statistics

# In[9]:


from keras.models import load_model
model= load_model('GRU_h5\gru_batchnormalization_holistic.h5')


# In[10]:


res = model.predict(X_test)


# In[11]:


yhat = model.predict(X_train)


# In[12]:


ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[13]:


multilabel_confusion_matrix(ytrue, yhat)


# In[14]:


accuracy_score(ytrue, yhat)


# In[15]:


yhat = model.predict(X_test)


# In[16]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[17]:


multilabel_confusion_matrix(ytrue, yhat)


# In[18]:


accuracy_score(ytrue, yhat)


# In[19]:


model.evaluate(X_test, y_test)


# In[20]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the saved model

classes=['my','father','help','me','now']
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[21]:


from sklearn.metrics import f1_score
print(f1_score(ytrue, yhat, average='macro'))
print(f1_score(ytrue, yhat,average='micro'))


# In[22]:


from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
import scikitplot as skplt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
predict_prob=model.predict([X_test])

#pred = model.predict(X_test)
y_pr = np.argmax(model.predict(X_test),axis=1).tolist()
y_tr = np.argmax(y_test, axis=1).tolist()
#y_pred = np.argmax(y_pred, axis=1).tolist()

precision = precision_score(y_tr,y_pr,average='macro')
recall = recall_score(y_tr, y_pr,average='macro')
 
print('Precision: ',precision)
print('Recall: ',recall)
 
#Plotting Precision-Recall Curve
precision = dict()
recall = dict()
for i in range(actions.shape[0]):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        predict_prob[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()


# In[23]:


from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_tr, y_pr)


# In[24]:


from sklearn.metrics import classification_report

print(classification_report(y_tr, y_pr)) 


# In[25]:


import seaborn as sns
ax=sns.heatmap(cf_matrix, annot=True)


# set y-axis label and ticks
ax.set_xlabel("predicted", fontsize=14, labelpad=20)
ax.set_ylabel("Actual", fontsize=14, labelpad=20)
ax.set_title("Confusion Matrix for ASL Recognation Model", fontsize=14, pad=20)


# In[26]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# In[ ]:




