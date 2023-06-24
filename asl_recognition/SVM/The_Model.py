#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2
import mediapipe as mp
import numpy as np
import os
import time
import seaborn as sns
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# In[3]:


# Define the hand gesture categories
gestures = ['smooking', 'not', 'good','i','drink','milk','she','always','lie','my','father','now']

# Define the directory where the hand gesture data is stored
data_dir = 'data13'

# Load the hand gesture data into memory
X = []
y = []
for gesture_idx, gesture in enumerate(gestures):
    gesture_dir = os.path.join(data_dir, gesture)
    for filename in os.listdir(gesture_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(gesture_dir, filename)
            landmark_list = np.loadtxt(filepath)
            X.append(landmark_list)
            y.append(gesture_idx)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Train a Support Vector Machine (SVM) classifier on the training data
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf.fit(X_train, y_train)

# Use the trained classifier to predict the labels of the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier on the test data
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the trained classifier to a file
model_filename = 'hand_classifier_1ast_yarab2.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(clf, f)


# In[2]:


model_filename = 'SVM_h5\hand_classifier_1ast_yarab2.pkl'
with open(model_filename, 'rb') as f:
    clf = pickle.load(f)


# In[6]:


y_pred = clf.predict(X_test)

# Print the predicted class labels


# In[8]:


cm = confusion_matrix(y_test, y_pred)


# In[9]:


print(cm)


# In[15]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)


# In[21]:


ax = sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=gestures, yticklabels=gestures)

# Set the axis labels and title
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')

# Show the plot
plt.show()


# In[26]:


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
predict_prob = clf.predict_proba(X_test)
precision = dict()
recall = dict()
for i in range(np.array(gestures).shape[0]):
    precision[i], recall[i], _ = precision_recall_curve(y_test,predict_prob[:, i], pos_label=i)
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()


# In[28]:


import seaborn as sns
ax=sns.heatmap(cm, annot=True)


# set y-axis label and ticks
ax.set_xlabel("predicted", fontsize=14, labelpad=20)
ax.set_ylabel("Actual", fontsize=14, labelpad=20)
ax.set_title("Confusion Matrix for ASL Recognation Model", fontsize=14, pad=20)


# In[31]:


sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%', cmap='Blues')


# In[ ]:




