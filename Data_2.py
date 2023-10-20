#This Code is to make the Dataset "three_parameter.csv"
#Which Created by the Calculating of the Images 
import cv2
import dlib
import csv
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist 

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

dict = {'56':3099, '62':3647, '67':3094, '72':3227, '77':1441}

LM_A = []
LM_B = []
LM_C = []
Class = []
eye_a_r = []

for P in dict:
    for k in range(0, int(dict[P]) + 1):
        image_path = "classification_frames/P10427"+str(P)+"_720/frame"+str(k)+".jpg"
        #os.chdir(image_path)
        image = cv2.imread(image_path)

        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            #print(img.shape)
        else:
            print('The specified path does NOT exist')

        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            def Eye_aspect_ratio(eye):
                A = dist.euclidean(eye[1], eye[5])
                LM_A.append(A)
                B = dist.euclidean(eye[2], eye[4])
                LM_B.append(B)
                C = dist.euclidean(eye[0], eye[3])
                LM_C.append(C)
                ear = (A + B) / (2.0 * C)
                eye_a_r.append(ear)
                return ear

            faces = detect(gray)

            for face in faces:
                shape = predict(gray, face)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

                left_eye = shape[42:48]
                right_eye = shape[36:42]

                left_ear = Eye_aspect_ratio(left_eye)
                right_ear = Eye_aspect_ratio(right_eye)

                EAR = (left_ear + right_ear) / 2.0

                eye_threshold = 0.3

                if EAR < eye_threshold:
                    Class.append("Drowsy")
                else:
                    Class.append("Alert")
            
            max_length = max(len(LM_A), len(LM_B), len(LM_C), len(Class))

            # Ensure that all lists have the same length by appending "Unknown" to Class if needed
            while len(LM_A) < max_length:
                LM_A.append("Unknown")

            while len(LM_B) < max_length:
                LM_B.append("Unknown")

            while len(LM_C) < max_length:
                LM_C.append("Unknown")

            while len(Class) < max_length:
                Class.append("Unknown")
        else:
            print("Error loading the image.")

        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
cv2.imshow("Drowsy Face Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows

print("LM_A:",len(LM_A))
print("LM_A:",len(LM_B))
print("LM_A:",len(LM_C))
print("Class:",len(Class))

df1 = {"Landmark_1":LM_A, "Landmark_2":LM_B, "Landmark_3":LM_C, "Class":Class}
df1 = pd.DataFrame(df1)
print(df1)

final_df = pd.concat([df1], ignore_index= True) 
final_df.to_csv("three_parameters.csv",index = False)