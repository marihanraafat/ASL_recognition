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
sequence_length = 20


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


from tensorflow.keras.layers import BatchNormalization, Dropout, Bidirectional, LSTM, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(20, 1662)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Use ReLU activation function in the output layer if the output is non-negative
if np.min(y_train) >= 0:
  model.add(Dense(len(actions), activation='relu'))
else:
  model.add(Dense(len(actions), activation='softmax'))

# Use early stopping and reduce learning rate on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Reduce the learning rate and add dropout layers
optimizer = tf.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.add(Dropout(0.2))
model.add(Dense(len(actions), activation='softmax'))

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=200, callbacks=[early_stop, reduce_lr])


# # predictions and statistics

# In[9]:


from keras.models import load_model
model= load_model('LSTM_h5\LSTM_OPTIMIZATION.h5')


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


# In[27]:


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


# In[28]:


from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_tr, y_pr)


# In[29]:


from sklearn.metrics import classification_report

print(classification_report(y_tr, y_pr)) 


# In[31]:


import seaborn as sns
ax=sns.heatmap(cf_matrix, annot=True)


# set y-axis label and ticks
ax.set_xlabel("predicted", fontsize=14, labelpad=20)
ax.set_ylabel("Actual", fontsize=14, labelpad=20)
ax.set_title("Confusion Matrix for ASL Recognation Model", fontsize=14, pad=20)


# In[32]:


sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# In[ ]:




