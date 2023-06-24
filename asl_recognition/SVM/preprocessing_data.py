#!/usr/bin/env python
# coding: utf-8

# # data collection

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Define the hand detection and pose estimation models
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the hand gesture categories
gestures = ['smooking', 'not', 'good','i','drink','milk','she','always','lie','my','father','now']

# Define the number of samples to collect for each gesture
num_samples = 100

# Define the time to wait between each frame in seconds
frame_wait_time = 7

# Create a directory for the hand gesture data
data_dir = 'data13'
os.makedirs(data_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)
time.sleep(frame_wait_time)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    for gesture in gestures:
        gesture_dir = os.path.join(data_dir, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        for i in range(num_samples):
            # Display the current frame number
            frame_text = f'Collecting {gesture} sample {i+1}/{num_samples}'
            print(frame_text)

            # Wait for the specified time between each frame
            #time.sleep(frame_wait_time)

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
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list = []
                for landmark in hand_landmarks.landmark:
                    landmark_list.append(landmark.x)
                    landmark_list.append(landmark.y)
                    landmark_list.append(landmark.z)

                # Save the landmark coordinates to a file
                filename = os.path.join(gesture_dir, f'{gesture}_{i}.txt')
                np.savetxt(filename, landmark_list)

                # Draw the detected keypoints on the hand
                annotated_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the frame with the current frame number and the detected keypoints
                cv2.putText(
                    annotated_frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Collecting Hand Gesture Data', annotated_frame)

            else:
                # If no hand is detected, display the frame with the current frame number
                cv2.putText(
                    frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Collecting Hand Gesture Data', frame)
            
            # Exit if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(frame_wait_time)
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

