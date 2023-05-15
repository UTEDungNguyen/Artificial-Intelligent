# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
from base64 import encode
import encodings
import tensorflow as tf
from tensorflow import keras
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import cv2  
import os
import cvlib as cv
import streamlit as st
                    
# load model
model = load_model('gender_detection.model')
age_model = load_model('Age_Detection.h5')

# open webcam
webcam = cv2.VideoCapture(0)
# webcam.set(3, 1280)
# webcam.set(4,720)
    
classes = ['man','woman']
age_classes = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-120']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        age_conf = age_model.predict(face_crop)[0]

        # get label with max accuracy
        idx = np.argmax(conf)
        age_idx = np.argmax(age_conf)

        label = classes[idx]
        age_label = list(reversed(age_classes))[age_idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        age_label = "{}: {:.2f}%".format(age_label, age_conf[age_idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y-16),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.putText(frame, age_label, (startX, Y+5),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("Gender and Age Detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()