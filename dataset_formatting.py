import json
import pandas as pd
import os

five_file = [1,2,3,4,5]
dataFrames = []

for i in five_file:
    json_file_path = f"B:\\ML\\Driver Fatigue\\driver_drowsy\\image_Datasets\\annotations_final_{i}.json"
    # Read the JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data).T

        # Flatten the "landmarks" column into separate columns
        landmarks_columns = [f"landmark_{i+1}" for i in range(5)]
        df[landmarks_columns] = df['landmarks'].apply(pd.Series)

        # Drop the original "landmarks" column
        df = df.drop(columns=['landmarks'])
        df.to_csv('image_folder_1.csv', index=False)
        df = pd.DataFrame(df)
        print(df.shape)

        dataFrames.append(df)

        print("CSV file 'output.csv' has been created.")
    else:
        print(f"File not found: {json_file_path}")

final_dataFrames = pd.concat(dataFrames, ignore_index= True)
final_dataFrames.to_csv("Driver_Fatigue_detect_image.csv",index = False)

df_final = pd.read_csv("Driver_Fatigue_detect_image.csv")
print(df_final)
print(df_final.shape)