mg_folder = [51,56,57,62,67,72,77]
folder_len = [3214,3099,3647,3094,3227,3104,1441]

import cv2
import dlib
import csv
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist 

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


dict = {'51':14, '56':9, '62':7, '67':4, '72':7, '77':1}

for P in dict:
    for k in range(1,int(dict[P])+1):
        image_path = f"classification_frames\P10427"+str(P)+"_720/frame"+str(k)+".jpg"
        print(image_path)
