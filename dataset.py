import cv2
import dlib
import csv
import numpy as np
from scipy.spatial import distance as dist

detect  = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("driver_drowsy\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

image_path = "WIN_20230506_20_20_42_Pro.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def Eye_Aspect_Ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

Eye_Threshhold = 0.3
Eye_Consec_fra = 48
counter = 0
TOTAL = 0

csv_filename = "drowsiness_results.csv"

with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Drowsy"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for facial landmark detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detect(gray)

        for face in faces:
            shape = predict(gray, face)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            # Extract the left and right eye landmarks
            left_eye = shape[42:48]
            right_eye = shape[36:42]

            # Calculate the EAR for both eyes
            left_ear = Eye_Aspect_Ratio(left_eye)
            right_ear = Eye_Aspect_Ratio(right_eye)

            # Average the EAR of both eyes
            ear = (left_ear + right_ear) / 2.0

            # Draw the face and eye regions
            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)

            # Check if the EAR is below the threshold
            if ear < Eye_Threshhold:
                counter += 1
                drowsy = 1  # Indicates drowsiness
            else:
                if counter >= Eye_Consec_fra:
                    TOTAL += 1
                counter = 0
                drowsy = 0  # Indicates not drowsy

            # Write the frame number and drowsy status to the CSV file
            csv_writer.writerow([TOTAL, drowsy])

        cv2.putText(frame, f"Drowsiness: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ... (rest of the code remains the same)
        cv2.putText(frame, f"Drowsiness: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsy Face Detection", frame)
    cv2.waitKey(0)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
    #key = cv2.waitKey(1)
    #if key == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
