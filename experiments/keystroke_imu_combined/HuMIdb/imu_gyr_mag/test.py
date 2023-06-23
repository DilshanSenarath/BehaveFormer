import subprocess
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import numpy as np
from  model import Model, PositionalEncoding, Transformer, TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
sys.path.append(str((Path(__file__)/"../../../../../utils").resolve()))
sys.path.append(str((Path(__file__)/"../../../../../evaluation").resolve()))
from Config import Config
from metrics import Metric


def convert_to_float(data):
    for user in data:
        for ses in user:
            for seqs in ses:
                if (seqs[0].dtype != "float64"):
                    seqs[0] = seqs[0].astype("float64")
                if (seqs[1].dtype != "float64"):
                    seqs[1] = seqs[1].astype("float64")


def scale(data):
    for user in data:
        for session in user:
            for i in range(len(session)):
                # Keystroke scaling
                for j in range(4):
                    if (j == 3):
                        session[i][0][:, j] = session[i][0][:, j] / 255
                    elif (j in [1,2]):
                        session[i][0][:, j] = session[i][0][:, j] / 1000
                
                # Remove very large values from fft values
                for j in [16, 17, 27, 28, 29]:
                    for k in range(100):
                        if (session[i][1][k][j] >= 1000000):
                            session[i][1][k][j] = 0.0
                        
                # IMU scaling
                for j in range(36):
                    if (j in [0,1,2]):
                        session[i][1][:, j] = session[i][1][:, j] / 10
                    elif (j in [3,4,5,24,25,26,27,28,29]):
                        session[i][1][:, j] = session[i][1][:, j] / 1000
                    elif (j in [15,16,17]):
                        session[i][1][:, j] = session[i][1][:, j] / 1000


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
        
        temp = self.eval_data[user_idx][session_idx][seq_idx][:]
        temp[0] = temp[0][:, [1,2,3]]

        temp[1] = temp[1][:, 12:36]

        return temp

def get_evaluate_results(feature_embeddings, num_enroll_sessions):
    _acc = []
    _usability = []
    _tcr = []
    _fawi = []
    _frwi = []
    for i in range(feature_embeddings.shape[0]):
        all_ver_embeddings = torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
        scores = Metric.cal_session_distance_humi(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
        periods = get_periods(i)
        labels = torch.tensor([1] * 2 + [0] * (feature_embeddings.shape[0] - 1) * 2)
        acc, threshold = Metric.eer_compute(scores[:2], scores[2:])
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
        seq = seqs[0]
        start = seq[0][0][0]
        end = seq[0][-1][0]
        i = -1
        while (end == 0):
            end = seq[0][i-1][0]
            i = i - 1
        return (end - start) / 1000

    periods = []
    for j in range(2):
        periods.append(get_window_time(testing_data[user_id][3 + j]))
    for i in range(len(testing_data)):
        if (i != user_id):
            for j in range(2):
                periods.append(get_window_time(testing_data[i][3 + j]))

    return periods


if __name__ == "__main__":
    config_data = Config().get_config_dict()
    preprocessed_data = config_data['preprocessed_data']
    
    subprocess.run(f'gdown {preprocessed_data["humi"]["test"]}', shell=True)

    if(config_data["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    infile = open("testing_keystroke_imu_data_all.pickle",'rb')
    testing_data = pickle.load(infile)
    infile.close()

    convert_to_float(testing_data)

    for user in testing_data:
        for idx, session in enumerate(user):
            user[idx] = session[:1]

    scale(testing_data)

    hyperparams = config_data['hyperparams']

    batch_size = hyperparams['batch_size']['humi']
    trg_len = hyperparams['target_len']
    number_of_enrollment_sessions = hyperparams['number_of_enrollment_sessions']['humi']
    num_verify_sessions = hyperparams['number_of_verify_sessions']['humi']

    results_path = f"{str((Path(__file__)/'../').resolve())}/results"
    best_model_save_path = f"{str((Path(__file__)/'../').resolve())}/best_models"

    subprocess.run(f"mkdir {results_path}", shell=True)

    if (config_data["GPU"] == "True"):
        test_model = torch.load(f"{best_model_save_path}/{sys.argv[2]}")
    else:
        test_model = torch.load(f"{best_model_save_path}/{sys.argv[2]}", map_location=torch.device('cpu'))

    test_model.train(False)

    t_dataset = TestDataset(testing_data)
    t_dataloader = DataLoader(t_dataset, batch_size=batch_size)

    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(t_dataloader):
            feature_embeddings.append(test_model([item[0].float(), item[1].float()]))

    if (sys.argv[1] == "basic"):
        res = get_evaluate_results(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions)
        pd.DataFrame([list(res)], columns=["eer", "usability", "tcr", "fawi", "frwi"]).to_csv(f"{results_path}/basic.csv")
    elif (sys.argv[1] == "det"):
        Metric.save_DET_curve(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions, num_verify_sessions, "humi", results_path)
    elif (sys.argv[1] == "pca"):
        Metric.save_PCA_curve(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions + num_verify_sessions, 10, results_path)