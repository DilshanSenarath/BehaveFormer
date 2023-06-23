import json    
from pathlib import Path
import pandas as pd
import os
import pickle
import re
import numpy as np
import zipfile
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as mp

import sys
sys.path.append(str((Path(__file__)/"../../../utils").resolve()))
from Config import Config

def extract_zip(zippedFile, toFolder):
    # Unzip a zip file and its contents, including nested zip files
    with zipfile.ZipFile(zippedFile, 'r') as zfile:
        zfile.extractall(path=toFolder)
    for filePath in zipfile.ZipFile(zippedFile).namelist():
        if re.search(r'\.zip$', filePath):
            completePath = os.path.join(toFolder, filePath)
            extract_zip(completePath, os.path.dirname(completePath))

def extract(absolute_path):
    dataset_location = Path(absolute_path)
    extract_location = dataset_location.parent/"dataset"
    print("Extracting...🚀")
    extract_zip(dataset_location, extract_location)
    print(f"File is unzipped in {extract_location} folder✅")
    return extract_location

def get_filtered_users():
    users = os.listdir("dataset/HuMI")
    user_set = set()

    for user in users:
        count = 0 
        if ("." not in user):
            sessions = os.listdir(f"dataset/HuMI/{user}")
            for session in sessions:
                if ("." not in session and "__" not in session):
                    key = pd.read_csv(f"dataset/HuMI/{user}/{session}/KEYSTROKE/key_data.csv", header=None, usecols=[0], names=["event_time"])
                    key[["event_time", "input_field", "key_code"]] = key["event_time"].str.split(' ', expand=True)
                    # indexField = key[(key["input_field"] == 'N') | (key["input_field"] == 'S') | (key["input_field"] == 'A')].index
                    # print(len(indexField))
                    # key.drop(indexField , inplace=True)
                    # key = key.sort_index().reset_index(drop=True)
                    if(key.shape[0]>0):
                        count = count + 1
            if(count==5):
                user_set.add(user)

    return list(user_set)

def keystroke_feature_extract(keystroke):
    #print("inside")
    if (keystroke.isnull().values.any()):
        print("WARNING: Original keystroke datframe contains NaN")
        keystroke.replace(np.nan, 0, inplace=True)
    keystroke["event_time"] = keystroke["event_time"].astype(int)
    keystroke["key_code"] = keystroke["key_code"].astype(int)
    for i in range(keystroke.shape[0]):
        #keystroke["event_time"] = keystroke["event_time"].astype(int)
        event_time = keystroke.iloc[i]["event_time"]
        key = keystroke.iloc[i]["key_code"]
        di_uu = 0.0
        tri_uu = 0.0

        if (i < keystroke.shape[0] - 1):
            di_uu = int(keystroke.iloc[i+1]["event_time"]) - int(keystroke.iloc[i]["event_time"])

        if (i < keystroke.shape[0] - 2):
            tri_uu = int(keystroke.iloc[i+2]["event_time"]) - int(keystroke.iloc[i]["event_time"])

        keystroke.at[i, ["event_time", "di_uu", "tri_uu", "key"]] = [event_time, di_uu, tri_uu, key]
        #print("over")
    return keystroke

def imu_feature_extract(imu_type_data):
    imu_type_data["event_time"] = imu_type_data["event_time"].astype(int)
    if (imu_type_data.isnull().values.any()):
        print("WARNING: IMU datframe contains NaN")
    
    imu_type_data["x"] = imu_type_data["x"].astype(float)
    imu_type_data["y"] = imu_type_data["y"].astype(float)
    imu_type_data["z"] = imu_type_data["z"].astype(float)

    imu_type_data["fft_x"] = np.abs(np.fft.fft(imu_type_data["x"].values))
    imu_type_data["fft_y"] = np.abs(np.fft.fft(imu_type_data["y"].values))
    imu_type_data["fft_z"] = np.abs(np.fft.fft(imu_type_data["z"].values))

    imu_type_data["fd_x"] = np.gradient(imu_type_data["x"].values, edge_order=2)
    imu_type_data["fd_y"] = np.gradient(imu_type_data["y"].values, edge_order=2)
    imu_type_data["fd_z"] = np.gradient(imu_type_data["z"].values, edge_order=2)

    imu_type_data["sd_x"] = np.gradient(imu_type_data["fd_x"].values, edge_order=2)
    imu_type_data["sd_y"] = np.gradient(imu_type_data["fd_y"].values, edge_order=2)
    imu_type_data["sd_z"] = np.gradient(imu_type_data["fd_z"].values, edge_order=2)

    return imu_type_data

def scaling(dataframe):
    std_scaler = StandardScaler()
    columns_names = list(dataframe.columns)
    dataframe = std_scaler.fit_transform(dataframe.to_numpy())
    dataframe = pd.DataFrame(dataframe, columns=columns_names)
    return dataframe

def embed_zero_padding(sequence, sequence_length):
    sample_count = sequence.shape[0]
    missing_sample_count = sequence_length - sample_count
    new_items_df = pd.DataFrame([[0] * sequence.shape[1] for i in range(missing_sample_count)], columns=list(sequence.columns))
    sequence = pd.concat([sequence,new_items_df],axis=0)
    sequence.reset_index(inplace = True, drop = True)

    return sequence

def sync_imu_data(accelerometer_data, gyroscope_data, magnetometer_data, sync_period, imu_sequence_length):
    if (accelerometer_data.isnull().values.any()):
        print("WARNING: Original accelerometer_data datframe contains NaN")
        accelerometer_data.replace(np.nan, 0, inplace=True)
    if (gyroscope_data.isnull().values.any()):
        print("WARNING: Original gyroscope_data datframe contains NaN")
        gyroscope_data.replace(np.nan, 0, inplace=True)
    if (magnetometer_data.isnull().values.any()):
        print("WARNING: Original magnetometer_data datframe contains NaN")
        magnetometer_data.replace(np.nan, 0, inplace=True)

    imu_prefixes = ["a", "g", "m"]
    column_names = ["x", "y", "z", "fft_x", "fft_y", "fft_z", "fd_x", "fd_y", "fd_z", "sd_x", "sd_y", "sd_z"]
    columns = []
    for prefix in imu_prefixes:
        for name in column_names:
            columns.append(f"{prefix}_{name}")

    imu_sequence = pd.DataFrame(columns=columns)

    accelerometer_min = math.inf
    gyroscope_min = math.inf
    magnetometer_min = math.inf

    if accelerometer_data.shape[0] != 0:
        accelerometer_min = accelerometer_data.iloc[0]['event_time']

    if gyroscope_data.shape[0] != 0:
        gyroscope_min = gyroscope_data.iloc[0]['event_time']

    if magnetometer_data.shape[0] != 0:
        magnetometer_min = magnetometer_data.iloc[0]['event_time']
  
    lowest_time = min(accelerometer_min, gyroscope_min, magnetometer_min)

    accelerometer_max = - math.inf
    gyroscope_max = - math.inf
    magnetometer_max = - math.inf

    if accelerometer_data.shape[0] != 0:
        accelerometer_max = accelerometer_data.iloc[accelerometer_data.shape[0] - 1]['event_time'] 

    if gyroscope_data.shape[0] != 0:  
        gyroscope_max = gyroscope_data.iloc[gyroscope_data.shape[0] - 1]['event_time']

    if magnetometer_data.shape[0] != 0:
        magnetometer_max = magnetometer_data.iloc[magnetometer_data.shape[0] - 1]['event_time']

    highest_time = max(accelerometer_max, gyroscope_max, magnetometer_max)

    start_time = lowest_time
    end_time = start_time + sync_period

    while start_time < highest_time:
        relevant_accelerometer_data = accelerometer_data.loc[(accelerometer_data['event_time'] >= start_time) & (accelerometer_data['event_time'] <= end_time)]
        relevant_gyroscope_data = gyroscope_data.loc[(gyroscope_data['event_time'] >= start_time) & (gyroscope_data['event_time'] <= end_time)]
        relevant_magnetometer_data = magnetometer_data.loc[(magnetometer_data['event_time'] >= start_time) & (magnetometer_data['event_time'] <= end_time)]

        if (relevant_accelerometer_data.shape[0] == 0 or relevant_gyroscope_data.shape[0] == 0 or relevant_magnetometer_data.shape[0] == 0):
            print("WARNING: Within sync period there is no elements")

        data = []
        for prefix in imu_prefixes:
            for name in column_names:
                if (prefix == "a"):
                    value = relevant_accelerometer_data[name].mean()
                elif (prefix == "g"):
                    value = relevant_gyroscope_data[name].mean()
                else:
                    value = relevant_magnetometer_data[name].mean()

                if math.isnan(value):
                    value = 0.0

                data.append(value)

        imu_sequence.loc[imu_sequence.shape[0]] = data

        start_time = start_time + sync_period
        end_time = end_time + sync_period

    if(imu_sequence.shape[0] > imu_sequence_length):
        imu_sequence = imu_sequence.head(imu_sequence_length)
    elif (imu_sequence.shape[0] < imu_sequence_length):
        imu_sequence = embed_zero_padding(imu_sequence, imu_sequence_length)

    return imu_sequence

def pre_process(event_data, event_sequence_length, imu_sequence_length, offset, accelerometer_data, gyroscope_data, magnetometer_data):
    length = event_data.shape[0]
    start = 0
    end = start + event_sequence_length
    event_sequences = []
    while start < length:
        if end >= length:
            event_sequences.append(event_data.loc[start: , :])
            break

        sequence = event_data.loc[start:(end - 1), :]
        sequence.reset_index(inplace = True, drop = True)
        event_sequences.append(sequence) 
        start = start + offset
        end = start + event_sequence_length

    imu_sequences = []

    max_imu_sample_count = -math.inf
  
    for sequence in event_sequences:
        start_time = int(sequence.iloc[0]['event_time'])
        end_time = int(sequence.iloc[-1]['event_time'])
        
        relevant_accelerometer_data = accelerometer_data.loc[(accelerometer_data['event_time'] >= start_time) & (accelerometer_data['event_time'] <= end_time)]
        relevant_gyroscope_data = gyroscope_data.loc[(gyroscope_data['event_time'] >= start_time) & (gyroscope_data['event_time'] <= end_time)]
        relevant_magnetometer_data = magnetometer_data.loc[(magnetometer_data['event_time'] >= start_time) & (magnetometer_data['event_time'] <= end_time)]

        sync_period = (end_time - start_time) / imu_sequence_length

        imu_sequence = sync_imu_data(relevant_accelerometer_data, relevant_gyroscope_data, relevant_magnetometer_data, sync_period, imu_sequence_length)

        imu_sequences.append(imu_sequence)
  
    event_sequences[len(event_sequences) - 1] = embed_zero_padding(event_sequences[len(event_sequences) - 1], event_sequence_length)

    return event_sequences, imu_sequences

def read_keystroke(absolute_path, users_list):
    DATASET_HOME_DIR = Path(absolute_path)
    dataset_complete_path = DATASET_HOME_DIR / "HuMI"

    all_keystroke_data = []
    user_count = 1
    for userid in users_list:
        session_data = []
        for session in os.listdir(dataset_complete_path/f"{userid}"):
            if ("." not in session):
                keystroke_csv_data = pd.read_csv(dataset_complete_path/f"{userid}/{session}/KEYSTROKE/key_data.csv", header=None, usecols=[0], names=["event_time"])
                keystroke_csv_data[["event_time", "input_field", "key_code"]] = keystroke_csv_data["event_time"].str.split(' ', expand=True)
                if (keystroke_csv_data.shape[0] != 0):
                    keystroke_csv_data["user_id"] = f"user_{userid}" 
                    keystroke_csv_data.drop(columns = ['user_id'], inplace=True)
                    keystroke_csv_data.drop(columns = ['input_field'], inplace=True)
                    
                    accelerometer_csv_data = pd.read_csv(dataset_complete_path/f"{userid}/{session}/KEYSTROKE/SENSORS/sensor_lacc.csv", header=None, usecols=[0], names=["event_time"])
                    if (accelerometer_csv_data.shape[0] == 0):
                        accelerometer_csv_data[["event_time"]] = [["0 0 0 0 0"], ["0 0 0 0 0"], ["0 0 0 0 0"]]
                    accelerometer_csv_data[["event_time", "orientation", "x", "y", "z"]] = accelerometer_csv_data["event_time"].str.split(' ', expand=True)
                    accelerometer_csv_data.drop(columns = ["orientation"], inplace=True)

                    gyroscope_csv_data = pd.read_csv(dataset_complete_path/f"{userid}/{session}/KEYSTROKE/SENSORS/sensor_gyro.csv", header=None, usecols=[0], names=["event_time"])
                    if (gyroscope_csv_data.shape[0] == 0):
                        gyroscope_csv_data[["event_time"]] = [["0 0 0 0 0"], ["0 0 0 0 0"], ["0 0 0 0 0"]]
                    gyroscope_csv_data[["event_time", "orientation", "x", "y", "z"]] = gyroscope_csv_data["event_time"].str.split(' ', expand=True)
                    gyroscope_csv_data.drop(columns = ["orientation"], inplace=True)
                    
                    magnetometer_csv_data = pd.read_csv(dataset_complete_path/f"{userid}/{session}/KEYSTROKE/SENSORS/sensor_magn.csv", header=None, usecols=[0], names=["event_time"])
                    if (magnetometer_csv_data.shape[0] == 0):
                        magnetometer_csv_data[["event_time"]] = [["0 0 0 0 0"], ["0 0 0 0 0"], ["0 0 0 0 0"]]
                    magnetometer_csv_data[["event_time", "orientation", "x", "y", "z"]] = magnetometer_csv_data["event_time"].str.split(' ', expand=True)
                    magnetometer_csv_data.drop(columns = ["orientation"], inplace=True)

                    keystroke_csv_data = keystroke_feature_extract(keystroke_csv_data)
                    accelerometer_csv_data = imu_feature_extract(accelerometer_csv_data)
                    gyroscope_csv_data = imu_feature_extract(gyroscope_csv_data)
                    magnetometer_csv_data = imu_feature_extract(magnetometer_csv_data)

                    keystroke_sequences, imu_sequences = pre_process(keystroke_csv_data, keystroke_sequence_len, imu_sequence_len, windowing_offset, accelerometer_csv_data, gyroscope_csv_data, magnetometer_csv_data)
                    
                    sequence_data = []
                    for i in range(len(keystroke_sequences)):
                        temp_keystroke = keystroke_sequences[i]
                        temp_imu = imu_sequences[i]
                        temp_keystroke = temp_keystroke.drop(columns=["key_code"])
                        sequence_data.append([temp_keystroke.to_numpy(), temp_imu.to_numpy()])

                    session_data.append(sequence_data)

                    print(f"INFO: Session {session} completed")

        all_keystroke_data.append(session_data)
        print(f"INFO: User {userid} completed ({user_count})")
        user_count = user_count + 1
                        
                    
    return all_keystroke_data

if __name__ == "__main__":
    config_data = Config().get_config_dict()["data"]

    # Dataset download url (You can generate the link from this site: https://hmog-dataset.github.io/hmog/)
    #dataset_url = "http://atvs.ii.uam.es/atvs/intranet/free_DB/HuMIDB/HuMI.zip"
    dataset_url = config_data["humi"]["dataset_url"]
    username = config_data["humi"]["username"]
    password = config_data["humi"]["password"]

    keystroke_sequence_len = 50 if config_data["keystroke_sequence_len"] is None else config_data["keystroke_sequence_len"]
    imu_sequence_len = 100 if config_data["imu_sequence_len"] is None else config_data["imu_sequence_len"]
    windowing_offset = 1 if config_data["humi"]["windowing_offset"] is None else config_data["humi"]["windowing_offset"]

    # If download_dataset is True then the dataset will be downloaded from the dataset_url
    download_dataset = True

    # Whether you want to extract the dataset
    extract_dataset = True

    # Whether you want to save the generated data files into google drive
    save_in_google_drive = True
    
    if (download_dataset):
        status = os.system(f"wget {dataset_url} --user={username} --password={password}")
        if (status != 0):
            print("Having an issue to download the dataset.")
        else:
            print("Download completed!")

    if (extract_dataset):
        extract("/content/HuMI.zip")

    user_list = get_filtered_users()

    training_user_list, val_test_user_list = train_test_split(user_list, test_size=130, train_size=298, shuffle=True)
    validation_user_list, testing_user_list = train_test_split(val_test_user_list, test_size=65, train_size=65, shuffle=True)

    training_keystroke_imu_data = read_keystroke("/content/dataset", training_user_list)
    outfile = open(f"training_keystroke_imu_data_all.pickle",'wb')
    pickle.dump(training_keystroke_imu_data, outfile)
    outfile.close()
    os.system(f"cp /content/training_keystroke_imu_data_all.pickle /content/drive/MyDrive/HuMI_Dataset/")

    validation_keystroke_imu_data = read_keystroke("/content/dataset", validation_user_list)
    outfile = open("validation_keystroke_imu_data_all.pickle",'wb')
    pickle.dump(validation_keystroke_imu_data, outfile)
    outfile.close()
    os.system(f"cp /content/validation_keystroke_imu_data_all.pickle /content/drive/MyDrive/HuMI_Dataset/")

    testing_keystroke_imu_data = read_keystroke("/content/dataset", testing_user_list)
    outfile = open("testing_keystroke_imu_data_all.pickle",'wb')
    pickle.dump(testing_keystroke_imu_data, outfile)
    outfile.close()
    os.system(f"cp /content/testing_keystroke_imu_data_all.pickle /content/drive/MyDrive/HuMI_Dataset/")