import cv2
import dlib
import csv
import numpy as np
from scipy.spatial import distance as dist 

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("driver_drowsy\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

image_path = "WIN_20230506_20_20_42_Pro.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def Eye_aspect_ratio(eye):
    A =  dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

faces = detect(gray)

for face in faces:
    shape = predict(gray,face)
    shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    left_eye = shape[42:48]
    right_eye = shape[36:42]

    left_ear = Eye_aspect_ratio(left_eye)
    riight_ear = Eye_aspect_ratio(right_eye)

    EAR = (left_ear + right_eye) / 2.0

    eye_threshold = 0.3

    if (EAR < eye_threshold).all():
        print("Drowsy")
    else:
        print("Normal")

cv2.imshow("Drowsy Face Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows