#!/usr/bin/env python
# coding: utf-8

# # data augmentation

# In[ ]:


from moviepy.editor import *
from moviepy.editor import VideoFileClip, vfx
from skimage.filters import gaussian
from moviepy.editor import VideoFileClip
import os
import numpy as np

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


# In[ ]:


def rotate(path,video,x):
    # loading video gfg
    clip = VideoFileClip(path+'\\'+video+'.mp4')
    clip = clip.rotate(x)
    clip.write_videofile(path+'\\'+video+"_rotate"+".mp4", fps=clip.fps)


# In[ ]:


def speed(path,video,x):
    # loading video gfg
    clip = VideoFileClip(path+'\\'+video+'.mp4')

    clip = clip.speedx(x)
    clip.write_videofile(path+'\\'+video+"_speed"+".mp4", fps=clip.fps)


# In[ ]:


def flip(path,video):   
    clip = VideoFileClip(path+'\\'+video+'.mp4')
    reversed_clip = clip.fx(vfx.mirror_x)
    reversed_clip.write_videofile(path+'\\'+video+"_flip"+".mp4", fps=clip.fps)


# In[ ]:


def bright(path,video):    
    clip = VideoFileClip(path+'\\'+video+'.mp4')
    clip=clip.fx(vfx.colorx, 1.2)  # 20% brighter
    clip=clip.fx(vfx.lum_contrast, 0, 40, 127)
    clip.write_videofile(path+'\\'+video+"_bright"+".mp4", fps=clip.fps)


# In[ ]:


def low_contrast(path,video,x):
    clip = VideoFileClip(path+'\\'+video+'.mp4')
    clip=clip.fx(vfx.lum_contrast, -x)
    clip.write_videofile(path+'\\'+video+"_low_contrast"+".mp4", fps=clip.fps)


# In[ ]:


def high_contrast(path,video,x):
    clip = VideoFileClip(path+'\\'+video+'.mp4')
    clip=clip.fx(vfx.lum_contrast,x)
    clip.write_videofile(path+'\\'+video+"_high_contrast"+".mp4", fps=clip.fps)


# In[ ]:


def pluring(path,video):

    def blur(image):
        """ Returns a blurred (radius=2 pixels) version of the image """
        return gaussian(image.astype(float), sigma=1.2)

    clip = VideoFileClip(path+'\\'+video+'.mp4')
    clip_blurred = clip.fl_image( blur )
    clip_blurred.write_videofile(path+'\\'+video+"_bluring"+".mp4", fps=clip.fps)


# In[ ]:


def crop(path,video,x):
    clip = VideoFileClip(path+'\\'+str(video)+'.mp4')
    clip=clip.crop(x)
    clip.write_videofile(path+'\\'+str(video)+"_crop"+".mp4", fps=clip.fps)


# In[ ]:


def all_augmentations(path,video):
    
    rotate=np.random.randint(2,6)
    speed='{:.2f}'.format(round(np.random.uniform(.6,.9),10))
    contrast=np.random.randint(20,40)
    crop=np.random.randint(60,80)
    
    clip = VideoFileClip(path+'\\'+video+'.mp4')
    clip = clip.speedx(float(speed))
    clip = clip.rotate(rotate)
    clip = clip.fx(vfx.mirror_x)
    clip=clip.fx(vfx.lum_contrast, -contrast)
    #clip=clip.fx(vfx.lum_contrast,contrast)
    clip=clip.crop(crop)
    clip.ipython_display()
    clip.write_videofile(path+'\\'+'all_'+video+'.mp4')


# In[ ]:


path=r'GP-project\GP-project\data_augmentation' # folder path that have vedios needs augmentation 
actions = np.array(['my','father','help','me','now'])
for action in actions:
    for video in os.listdir(path+'\\'+str(action)):
        pluring(path+'\\'+str(action),video.replace('.mp4',''))
        crop(path+'\\'+str(action),video.replace('.mp4',''),66)
        high_contrast(path+'\\'+str(action),video.replace('.mp4',''),30)
        low_contrast(path+'\\'+str(action),video.replace('.mp4',''),30)
        bright(path+'\\'+str(action),video.replace('.mp4',''))
        flip(path+'\\'+str(action),video.replace('.mp4',''))
        speed(path+'\\'+str(action),video.replace('.mp4',''),.8)
        rotate(path+'\\'+str(action),video.replace('.mp4',''),3)


# In[ ]:





# # data collection 

# In[ ]:


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# In[ ]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


# In[ ]:


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


# In[ ]:


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


# In[ ]:


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# # collect kepoints from videos
# 

# In[ ]:


# make directory called points_test have 200 folder empty for 200 vedios 
import os
newpath = "GP-project/points_test"
words_extraction=['my','father','help','me','now']
if not os.path.exists(newpath):
    os.makedirs(newpath)
    for i in(words_extraction):
        os.makedirs(newpath+"/"+str(i))

for i in(words_extraction):
    for j in range(200):#number of vedios
        os.makedirs('GP-project/points_test'+"/"+str(i)+"/"+str(j))


# In[ ]:


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('GP-project/points_test') 

# Actions that we try to detect
actions = np.array(['my','father','help','me','now'])

# 200 videos worth of data
no_sequences = 200

# Videos are going to be 20 frames in length
sequence_length = 20


# In[ ]:


import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('GP-project'+'/'+'points_test') 
#path="GP-project/video_datasets"
# Actions that we try to detect
actions = np.array(['father','help','me','now','my'])
path="GP-project\GP-project\data_augmentation"
# 200 videos worth of data
seq=200
no_sequences = range(seq)

# Videos are going to be 21 frames in length
sequence_length = 20

#❤❤❤❤ i did it ❤❤❤❤#

var_1=0 
var_2=actions[0]
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    for action in actions:
        
        for video_id in os.listdir(path+'/'+str(action)):
         
            for sequence in no_sequences:
                cap = cv2.VideoCapture(path+'/'+action+'/'+video_id)
               
                for frame_num in range(sequence_length):
                    
                    ret, frame = cap.read()
                    if ret:
                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)   

                        keypoints = extract_keypoints(results)
                        
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)
                        print(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        print(npy_path)
                        print(video_id)
                        
                        if cap.get(cv2.CAP_PROP_POS_FRAMES)==sequence_length+1:
                            cap.release()
                            cv2.destroyAllWindows()
                            break
                                     
                    else:
                        break
                       
                var_1+=1    
                if var_1==sequence+1:
                    #print("this is the var 1",var_1)
                    no_sequences=range(var_1,seq) 
                    break
                      
            if var_1==seq:
                no_sequences=range(seq) 
                var_1=0


# # return frams number of a specific video

# In[ ]:


def return_frams_number(action): #frams number for each vedio
    path="GP-project\GP-project\data_augmentation"#GP-project\GP-project\kaggle+wlasl
    frams_list=[]
    for video_id in os.listdir(path+'/'+str(action)):
        cap = cv2.VideoCapture(path+'/'+action+'/'+video_id)
        frams_list.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))==18:
            print(path+'/'+action+'/'+video_id)
    return frams_list    
min(return_frams_number('we'))


# In[ ]:




