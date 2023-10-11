import dlib
import cv2
import numpy as np
import pandas as pd
import os

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"B:\ML\Driver Fatigue\driver_drowsy\shape_predictor_68_face_landmarks.dat")
#predictor = dlib.shape_predictor(r"B:\ML\Driver Fatigue\shape_predictor_68_face_landmarks.dat")


# Function to calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Function to detect and extract eye landmarks from an image
def detect_and_extract_eye_landmarks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    if len(faces) == 0:
        return None, None

    # Get facial landmarks for the first face found
    shape = predictor(gray, faces[0])
    
    # Extract eye landmarks
    left_eye_pts = [shape.part(i) for i in range(36, 42)]
    right_eye_pts = [shape.part(i) for i in range(42, 48)]

    return left_eye_pts, right_eye_pts

# Path to the directory containing your image dataset
image_dir = 'B:\ML\Driver Fatigue'

# Create a DataFrame to store the data
data = {
    "Image": [],
    "Left_Eye_Distance": [],
    "Right_Eye_Distance": [],
    "Label": []
}

# Loop through each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        
        left_eye_pts, right_eye_pts = detect_and_extract_eye_landmarks(image_path)

        if left_eye_pts is not None and right_eye_pts is not None:
            # Calculate Euclidean distances for left and right eyes
            left_eye_distance = euclidean_distance(left_eye_pts[0], left_eye_pts[3])
            right_eye_distance = euclidean_distance(right_eye_pts[0], right_eye_pts[3])

            # Determine if eyes are open or closed based on thresholds (you can adjust these)
            if left_eye_distance < 6.0 and right_eye_distance < 6.0:
                label = "Closed"
            else:
                label = "Open"

            # Append data to the DataFrame
            data["Image"].append(filename)
            data["Left_Eye_Distance"].append(left_eye_distance)
            data["Right_Eye_Distance"].append(right_eye_distance)
            data["Label"].append(label)

# Create a DataFrame and save it as a CSV file
df = pd.DataFrame(data)
df.to_csv("eye_distances.csv", index=False)

print("Data saved to eye_distances.csv")
    