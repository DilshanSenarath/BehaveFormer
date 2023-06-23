from pathlib import Path
import sys
sys.path.append(str((Path(__file__)/"../../../utils").resolve()))

import subprocess
import pandas as pd
from Config import Config

dataset_url = Config().get_config_dict()["data"]["aalto"]["dataset_url"]

subprocess.run("wget http://buildlogs-seed.centos.org/c7.1804.00.x86_64/unzip/20180411052347/6.0-19.el7.x86_64/unzip-6.0-19.el7.x86_64.rpm", shell=True)
subprocess.run("add-apt-repository universe", shell=True)
subprocess.run("apt-get install alien", shell=True)
subprocess.run("alien -i unzip-6.0-19.el7.x86_64.rpm", shell=True)
subprocess.run(f"wget {dataset_url}", shell=True)
subprocess.run("unzip csv_raw_and_processed.zip", shell=True)

headers = pd.read_csv("Data_Raw/test_sections_header.csv").Field.tolist()
test_sections = pd.read_csv("Data_Raw/test_sections.csv", names=headers, encoding='latin-1', on_bad_lines='skip')

test_sections = test_sections[['TEST_SECTION_ID', 'PARTICIPANT_ID']]

headers = pd.read_csv("Data_Raw/keystrokes_header.csv").Field.tolist()
data = pd.read_csv("Data_Raw/keystrokes.csv", names=headers, encoding='latin-1', chunksize=10000000, on_bad_lines='skip')
dfs = []
for df in data:
    dfs.append(df[["TEST_SECTION_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE"]].merge(test_sections))

keystroke = pd.concat(dfs)

keystroke.rename(columns = {'TEST_SECTION_ID':'session_id', 'PARTICIPANT_ID': 'user_id', 'PRESS_TIME':'press_time', 'RELEASE_TIME':'release_time', 'KEYCODE':'key_code'}, inplace = True)

keystroke.to_csv("keystroke_data.csv")

subprocess.run("cp -r keystroke_data.csv drive/MyDrive/")