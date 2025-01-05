import os
import cv2
import numpy as np
from utils import get_face_landmarks

# Set parent directory for data
data_dir = './data'

output = []

for emotion_indx, emotion in enumerate(os.listdir(data_dir)):

    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):

        image_path = os.path.join(data_dir, emotion, image_path_)
        image = cv2.imread(image_path)
        face_landmarks = get_face_landmarks(image) # will get 1404 landmarks

        if len(face_landmarks) == 1404:

            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))