import subprocess
from pathlib import Path
import sys
sys.path.append(str((Path(__file__)/"../../../utils").resolve()))
sys.path.append(str((Path(__file__)/"../../../evaluation").resolve()))
from Config import Config
import torch
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from metrics import Metric

def scale(data):
    for user in data:
        for session in user:
            for i in range(len(session)):
                # Keystroke scaling
                for j in range(10):
                    if (j == 9):
                        # hold latency, key code
                        session[i][0][:, j] = session[i][0][:, j] / 255
                    else:
                        session[i][0][:, j] = session[i][0][:, j] / 1000
                        
                # IMU scaling
                for j in range(36):
                    if (j == 0 or j == 1 or j == 2):
                        session[i][1][:, j] = session[i][1][:, j] / 10
                    elif (j == 3 or j == 4 or j == 5 or j == 15 or j == 16 or j == 17):
                        session[i][1][:, j] = session[i][1][:, j] / 1000
                    elif (j == 24 or j == 25 or j == 26):
                        session[i][1][:, j] = session[i][1][:, j] / 100
                    elif (j == 27 or j == 28 or j == 29):
                        session[i][1][:, j] = session[i][1][:, j] / 10000

        
# Dataset for validating/testing the model
class TestDataset(Dataset):
    def __init__(self, eval_data):
        self.eval_data = eval_data
        self.num_sessions = len(self.eval_data[0])
        self.num_seqs = len(self.eval_data[0][0])

    def __len__(self):
        return  math.ceil(len(self.eval_data) * self.num_sessions * self.num_seqs);

    def __getitem__(self, idx):
        t_session = idx // self.num_seqs
        user_idx = t_session // self.num_sessions
        session_idx = t_session % self.num_sessions
        seq_idx = idx % self.num_seqs

        return self.eval_data[user_idx][session_idx][seq_idx][0]
    
def get_evaluate_results(feature_embeddings, num_enroll_sessions):
    _acc = []
    _usability = []
    _tcr = []
    _fawi = []
    _frwi = []
    for i in range(feature_embeddings.shape[0]):
        all_ver_embeddings = torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
        scores = Metric.cal_session_distance_hmog(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
        periods = get_periods(i)
        labels = torch.tensor([1] * 5 + [0] * (feature_embeddings.shape[0] - 1) * 5)
        acc, threshold = Metric.eer_compute(scores[:5], scores[5:])
        usability = Metric.calculate_usability(scores, threshold, periods, labels)
        tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
        frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
        fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)
        _acc.append(acc)
        _usability.append(usability)
        _tcr.append(tcr)
        _fawi.append(fawi)
        _frwi.append(frwi)
        
    return 100 - np.mean(_acc, axis=0), np.mean(_usability, axis=0), np.mean(_tcr, axis=0), np.mean(_fawi, axis=0), np.mean(_frwi, axis=0) 
        
def get_periods(user_id):
    def get_window_time(seqs):
        time = 0
        for seq in seqs:
            if (time == 0):
                time = time + np.sum(seq[0], axis=0)[2] + seq[0][-1][0]
            else:
                time = time + np.sum(seq[0][-5:], axis=0)[2] + seq[0][-1][0]
        return time

    periods = []
    for j in range(5):
        periods.append(get_window_time(testing_data[user_id][3 + j]))
    for i in range(len(testing_data)):
        if (i != user_id):
            for j in range(5):
                periods.append(get_window_time(testing_data[i][3 + j]))

    return periods

if __name__ == "__main__":
    config=Config()
    data = config.get_config_dict()["data"]
    hyperparams = config.get_config_dict()["hyperparams"]

    test_id = config.get_config_dict()["preprocessed_data"]["hmog"]["test"]

    subprocess.run(f"gdown {test_id}", shell=True)

    if(config.get_config_dict()["GPU"]=="True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    infile = open("testing_keystroke_imu_data_all.pickle",'rb')
    testing_data = pickle.load(infile)
    infile.close()
    
    for user in testing_data:
        for idx, session in enumerate(user):
            user[idx] = session[:50]

    scale(testing_data)

    batch_size = hyperparams['batch_size']['hmog']
    trg_len = hyperparams['target_len']
    number_of_enrollment_sessions = hyperparams['number_of_enrollment_sessions']['hmog']
    number_of_verification_sessions = hyperparams['number_of_verify_sessions']['hmog']

    results_path = f"{str((Path(__file__)/'../').resolve())}/results"
    best_model_save_path = f"{str((Path(__file__)/'../').resolve())}/best_models"

    subprocess.run(f"mkdir {results_path}", shell=True)

    if (config.get_config_dict()["GPU"] == "True"):
        test_model = torch.load(f"{best_model_save_path}/{sys.argv[2]}")
    else:
        test_model = torch.load(f"{best_model_save_path}/{sys.argv[2]}", map_location=torch.device('cpu'))
    test_model.train(False)

    t_dataset = TestDataset(testing_data)
    t_dataloader = DataLoader(t_dataset, batch_size=batch_size)

    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(t_dataloader):
            feature_embeddings.append(test_model(item.float()))

    if (sys.argv[1] == "basic"):
        res = get_evaluate_results(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions)
        pd.DataFrame([list(res)], columns=["eer", "usability", "tcr", "fawi", "frwi"]).to_csv(f"{results_path}/basic.csv")
    elif (sys.argv[1] == "det"):
        Metric.save_DET_curve(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions, number_of_verification_sessions, "hmog", results_path)
    elif (sys.argv[1] == "pca"):
        Metric.save_PCA_curve(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions+number_of_verification_sessions, 10, results_path)





