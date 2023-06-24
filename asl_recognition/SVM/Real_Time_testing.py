#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import os
import time

import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


# In[2]:


def get_bounding_box(hand_landmarks, image):
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    return x_min, y_min, x_max, y_max


# In[3]:


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the hand gesture categories
gestures = ['smoking', 'not', 'good', 'i', 'drink', 'milk', 'she', 'always', 'lie', 'my', 'father', 'now']

# Load the trained classifier from file
model_filename = 'SVM_h5\hand_classifier_1ast_yarab2.pkl'
with open(model_filename, 'rb') as f:
    clf = pickle.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the hand detection and pose estimation models
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print('Failed to read frame')
            continue

        # Convert the frame to RGB for processing by MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the hand detection and pose estimation models
        results = hands.process(frame_rgb)
        
        # Extract the landmark coordinates from the hand landmarks
        landmark_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmark_list.append(landmark.x)
                    landmark_list.append(landmark.y)
                    landmark_list.append(landmark.z)

        # Use the trained classifier to predict the hand gesture
        gesture = None
        if landmark_list:
            gesture_idx = clf.predict([landmark_list])[0]
            gesture = gestures[gesture_idx]
            print(f'Predicted gesture: {gesture}')

        # Draw the detected keypoints on the hand
        annotated_frame = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                ################
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, frame)
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
                ############

        # Display the frame with the detected keypoints and predicted gesture
        if gesture is not None:
            cv2.putText(
                annotated_frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Hand Gesture Recognition', annotated_frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


# In[ ]:


cap.release()
cv2.destroyAllWindows()

