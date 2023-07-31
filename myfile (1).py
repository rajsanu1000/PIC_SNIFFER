import dlib
import sys
import cv2
import matplotlib.pyplot as plt
import face_recognition
import numpy as np
import os

from pathlib import Path
from os import listdir

cam = cv2.VideoCapture(0)

cv2.namedWindow("Python Webcam Screenshot App")

img_counter=0

while True:
    ret,frame=cam.read()
    if not ret:
        print("failed to grab frame")

        break

    cv2.imshow("test",frame)

    k=cv2.waitKey(1)

    if k%256==27:
        print("Escape hit,closing the app")
        break

    elif k%256==32:
        img_name="opencv_frame_{}.png".format(img_counter)

        print(type(img_name))
        cv2.imwrite(img_name,frame)

        print("screenshot taken")

        img_counter+=1




cam.release()

mobileno=input("Enter your mobile number ->")

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

folder_dir = 'persons'
images = Path(folder_dir).glob('*.png')

img1_path=img_name
for image in images:
    img2_path=str(image)
    img1=dlib.load_rgb_image(img1_path)
    img2=dlib.load_rgb_image(img2_path)


    img1_detected=detector(img1,1)
    img2_detected=detector(img2,1)


    img1_shape=sp(img1,img1_detected[0])
    img2_shape=sp(img2,img2_detected[0])


    img1_aligned=dlib.get_face_chip(img1,img1_shape)
    img2_aligned=dlib.get_face_chip(img2,img2_shape)


    img1_representation=model.compute_face_descriptor(img1_aligned)
    img2_representation=model.compute_face_descriptor(img2_aligned)

    img1_representation=np.array(img1_representation)
    img2_representation=np.array(img2_representation)

    
 

 
    distance = findEuclideanDistance(img1_representation, img2_representation)
    threshold = 0.6 #distance threshold declared in dlib docs for 99.38% confidence score on LFW data set
 
    if distance  <threshold:
        print("they are same")
        
        img2_path=img2_path.replace('\\','/')
        f_path = Path(img2_path)

        import pywhatkit as pwk


        pwk.sendwhats_image(mobileno, f_path)

    
    else:
        print("they are different")