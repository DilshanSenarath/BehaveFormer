import subprocess
from pathlib import Path
import sys
sys.path.append(str((Path(__file__)/"../../../../utils").resolve()))
sys.path.append(str((Path(__file__)/"../../../../evaluation").resolve()))
from Config import Config
import torch
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from  model import KeystrokeTransformer
import math
from metrics import Metric
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.style.use('seaborn-v0_8-bright')
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})
mpl.rc('axes',edgecolor='black')
mpl.rcParams.update({'font.size': 13})

def preprocess():
    # Load dataset
    data = pd.read_csv("keystroke_data.csv")

    # Ensure whether dataset doesn't have any NaN values
    assert(data.isnull().values.any() == False)

    # Create numpy arrays for each user session containing press_time, release_time, key_code as dictionary
    grouped = data.groupby("user_id")
    data_dict = {x: group for x,group in grouped}
    del data
    for user in data_dict:
        data_dict[user] = [group[["press_time", "release_time", "key_code"]].to_numpy() for x, group in data_dict[user].groupby("session_id")]

    # Remove users who don't have 15 sessions
    removing_users = []
    for user in data_dict:
        if (len(data_dict[user]) != 15):
            removing_users.append(user)
    for key in removing_users:
        data_dict.pop(key, None)

    training_data = list(data_dict.values())
    print(len(training_data))

    # Create training and testing user data lists
    training_data = list(data_dict.values())[:-1050]
    validation_data = list(data_dict.values())[-1050:-1000]
    testing_data = list(data_dict.values())[-1000:]
    del data_dict

    # Save testing dataset for further use
    outfile = open("testing_data.pickle",'wb')
    pickle.dump(testing_data, outfile)
    outfile.close()
    # Save training dataset for further use
    outfile = open("training_data.pickle",'wb')
    pickle.dump(training_data, outfile)
    outfile.close()
    # Save validation dataset for further use
    outfile = open("validation_data.pickle",'wb')
    pickle.dump(validation_data, outfile)
    outfile.close()
        
# Dataset for validating/testing the model
class TestDataset(Dataset):
    def __init__(self, eval_data, batch_size, seq_len):
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __len__(self):
        return math.ceil(len(self.eval_data) * 15);

    def __getitem__(self, idx):
        user_idx = idx // 15
        offset = idx % 15

        return self.pad_sequence(self.eval_data[user_idx][offset])

    def pad_sequence(self, sequence):
        if (len(sequence) == self.seq_len):
            return sequence
        elif (len(sequence) < self.seq_len):
            row_count = self.seq_len - len(sequence)
            return np.append(sequence, np.array([0.0] * 10 * row_count).reshape(row_count, 10), axis=0)
        else:
            return sequence[0:self.seq_len]

def extract_normalize_features(dataset):
    for user_sequences in dataset:
        for idx in range(len(user_sequences)):
            user_sequences[idx] = np.append(user_sequences[idx], np.reshape([0.0] * len(user_sequences[idx]) * 7, (len(user_sequences[idx]),7)), axis=1)
            sequence = user_sequences[idx]
            for i in range(len(sequence)):
                m = sequence[i][1] - sequence[i][0]
                ud = 0.0
                dd = 0.0
                uu = 0.0
                du = 0.0
                t_ud = 0.0
                t_dd = 0.0
                t_uu = 0.0
                t_du = 0.0
                if (i != len(sequence) - 1):
                    ud = sequence[i+1][0] - sequence[i][1]
                    dd = sequence[i+1][0] - sequence[i][0]
                    uu = sequence[i+1][1] - sequence[i][1]
                    du = sequence[i+1][1] - sequence[i][0]
                if (i < len(sequence) - 2):
                    t_ud = sequence[i+2][0] - sequence[i][1]
                    t_dd = sequence[i+2][0] - sequence[i][0]
                    t_uu = sequence[i+2][1] - sequence[i][1]
                    t_du = sequence[i+2][1] - sequence[i][0]
                key_code = sequence[i][2]

                sequence[i][0] = m / 1000
                sequence[i][1] = ud / 1000
                sequence[i][2] = dd / 1000
                sequence[i][3] = uu / 1000
                sequence[i][4] = du / 1000
                sequence[i][5] = t_ud / 1000
                sequence[i][6] = t_dd / 1000
                sequence[i][7] = t_uu / 1000
                sequence[i][8] = t_du / 1000
                sequence[i][9] = key_code / 255

def get_evaluate_results(feature_embeddings, num_enroll_sessions):
    _acc = []
    _usability = []
    _tcr = []
    _fawi = []
    _frwi = []
    for i in range(feature_embeddings.shape[0]):
        enroll_emb = torch.unsqueeze(feature_embeddings[i,:num_enroll_sessions], dim=0)
        verify_emb = torch.unsqueeze(torch.cat([feature_embeddings[i,10:], feature_embeddings[:i,10], feature_embeddings[i+1:,10]], dim=0), dim=1)
        
        scores = torch.mean(torch.linalg.norm(verify_emb-enroll_emb, dim=-1), dim=-1)
        periods = get_periods(i)
        labels = torch.tensor([1] * 5 + [0] * (feature_embeddings.shape[0] - 1))
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
    def get_window_time(sequence):
        return np.sum(sequence, axis=0)[2] + sequence[-1][0]

    periods = []
    for j in range(5):
        periods.append(get_window_time(testing_data[user_id][10 + j]))
    for i in range(len(testing_data)):
        if (i != user_id):
            periods.append(get_window_time(testing_data[i][10]))

    return periods

def save_DET_curve(feature_embeddings, num_enroll_sessions):
    values = 0
    div = 0
    _min = math.inf
    _max = - math.inf
    eer_positions = []
    
    for i in range(feature_embeddings.shape[0]):
        enroll_emb = torch.unsqueeze(feature_embeddings[i,:num_enroll_sessions], dim=0)
        verify_emb = torch.unsqueeze(torch.cat([feature_embeddings[i,10:], torch.flatten(feature_embeddings[:i,10:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,10:], start_dim=0, end_dim=1)], dim=0), dim=1)

        scores = torch.mean(torch.linalg.norm(verify_emb-enroll_emb, dim=-1), dim=-1)
        min_max = get_min_max(scores[:(15 - num_enroll_sessions)], scores[(15 - num_enroll_sessions):])
        if (min_max[0] < _min):
            _min = min_max[0]
        if (min_max[1] > _max):
            _max = min_max[1]
            
    for i in range(feature_embeddings.shape[0]):
        enroll_emb = torch.unsqueeze(feature_embeddings[i,:num_enroll_sessions], dim=0)
        verify_emb = torch.unsqueeze(torch.cat([feature_embeddings[i,10:], torch.flatten(feature_embeddings[:i,10:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,10:], start_dim=0, end_dim=1)], dim=0), dim=1)

        scores = torch.mean(torch.linalg.norm(verify_emb-enroll_emb, dim=-1), dim=-1)
        eer_pos = get_far_frr_summary(scores[:(15 - num_enroll_sessions)], scores[(15 - num_enroll_sessions):], _min, _max)
        eer_positions.append(eer_pos)
        
    fix_eer_pos = max(eer_positions)
    max_ele_count = fix_eer_pos - min(eer_positions)
    
    for i in range(feature_embeddings.shape[0]):
        enroll_emb = torch.unsqueeze(feature_embeddings[i,:num_enroll_sessions], dim=0)
        verify_emb = torch.unsqueeze(torch.cat([feature_embeddings[i,10:], torch.flatten(feature_embeddings[:i,10:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,10:], start_dim=0, end_dim=1)], dim=0), dim=1)

        scores = torch.mean(torch.linalg.norm(verify_emb-enroll_emb, dim=-1), dim=-1)
        far_frrs = get_far_frr(scores[:(15 - num_enroll_sessions)], scores[(15 - num_enroll_sessions):], _min, _max, fix_eer_pos, max_ele_count, eer_positions[i])
        if (type(values) == int):
            values = far_frrs[0]
            div = far_frrs[1]
        else:
            values = values + far_frrs[0]
            div = div + far_frrs[1]
            
    values = values / div
    
    df = pd.DataFrame(values, columns=["FAR", "FRR"])
    df.to_csv(f"{results_path}/far-frr.csv")

def get_min_max(scores_g, scores_i):
    ini=torch.min(torch.cat([scores_g, scores_i], dim=0)).item()
    fin=torch.max(torch.cat([scores_g, scores_i], dim=0)).item()

    return ini, fin

def get_far_frr_summary(scores_g, scores_i, ini, fin):
    far = []
    frr = []
    
    paso=(fin-ini)/10000
    threshold = ini-paso
    while threshold < fin+paso:
        far.append(torch.count_nonzero(scores_i >= threshold).item()/scores_i.shape[0])
        frr.append(torch.count_nonzero(scores_g < threshold).item()/scores_g.shape[0])
        threshold = threshold + paso
    
    gap = torch.abs(torch.tensor(far) - torch.tensor(frr))
    j = torch.nonzero(gap == torch.min(gap))
    index = j[0][0].item()

    return index

def get_far_frr(scores_g, scores_i, ini, fin, fix_eer_pos, max_ele_count, eer_pos):
    far_frr = []

    paso=(fin-ini)/10000
    threshold = ini-paso
    while threshold < fin+paso:
        far = torch.count_nonzero(scores_i >= threshold).item()/scores_i.shape[0]
        frr = torch.count_nonzero(scores_g < threshold).item()/scores_g.shape[0]
        far_frr.append([far, frr])
        threshold = threshold + paso
        
    front_add = []
    adding_count = fix_eer_pos - eer_pos
    for _ in range(adding_count):
        front_add.append([0.0,0.0])
    if (adding_count != 0):
        far_frr = front_add + far_frr[:-adding_count]
    for _ in range(len(far_frr) - adding_count):
        front_add.append([1.0,1.0])
        
    return np.array(far_frr), np.array(front_add)

def save_PCA_curve(feature_embeddings, num_enroll_sessions, number_of_users):
    users = []
    for i in range(10):
        val = np.random.randint(0, 1000)
        while (val in users):
            val = np.random.randint(0, 1000)
        users.append(val)
    values = TSNE(n_iter=1000, perplexity=14).fit_transform(feature_embeddings[users].flatten(start_dim=0, end_dim=1).cpu().numpy())
    
    labels = []
    for user in users:
        for i in range(15):
            labels.append(user)

    pd.DataFrame([[silhouette_score(values, labels)]], columns=["Silhouette Score"]).to_csv(f"{results_path}/silhouette_score.csv")
    
    df = pd.DataFrame(values, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
    labels = []
    for i in range(number_of_users):
        for j in range(15):
            labels.append(f"User {i+1}")
    df["Users"] = labels

    g = sns.relplot(
        data=df,
        x="t-SNE Dimension 1", y="t-SNE Dimension 2",
        hue="Users",
        sizes=(10, 200),
        palette=["red", "green", "blue", "black", "blueviolet", "orange", "grey", "brown", "deeppink", "purple"]
    )
    g.set(xscale="linear", yscale="linear")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)
    g._legend.remove()
    plt.savefig(f'{results_path}/pca_graph.png', dpi=400)

if __name__ == "__main__":
    config = Config()
    keystroke_data_id = config.get_config_dict()["preprocessed_data"]["aalto"]["keystroke"]

    subprocess.run(f"gdown {keystroke_data_id}", shell=True)

    if (config.get_config_dict()["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    preprocess()

    infile = open("testing_data.pickle",'rb')
    testing_data = pickle.load(infile)
    infile.close()

    extract_normalize_features(testing_data)

    batch_size = config.get_config_dict()["hyperparams"]["batch_size"]["aalto"]
    l = config.get_config_dict()["data"]["keystroke_sequence_len"]

    results_path = f"{str((Path(__file__)/'../').resolve())}/results"
    best_model_save_path = f"{str((Path(__file__)/'../').resolve())}/best_models"

    subprocess.run(f"mkdir {results_path}", shell=True)

    if (config.get_config_dict()["GPU"] == "True"):
        test_model = torch.load(f"{best_model_save_path}/{sys.argv[2]}")
    else:
        test_model = torch.load(f"{best_model_save_path}/{sys.argv[2]}", map_location=torch.device('cpu'))
    test_model.train(False)

    t_dataset = TestDataset(testing_data, batch_size, l)
    t_dataloader = DataLoader(t_dataset, batch_size=batch_size)
    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(t_dataloader):
            feature_embeddings.append(test_model(item.float()))

    if (sys.argv[1] == "basic"):
        res = get_evaluate_results(torch.cat(feature_embeddings, dim=0).view(len(testing_data),15, 64), 10)
        pd.DataFrame([list(res)], columns=["eer", "usability", "tcr", "fawi", "frwi"]).to_csv(f"{results_path}/basic.csv")
    elif (sys.argv[1] == "det"):
        save_DET_curve(torch.cat(feature_embeddings, dim=0).view(len(testing_data),15, 64), 10)
    elif (sys.argv[1] == "pca"):
        save_PCA_curve(torch.cat(feature_embeddings, dim=0).view(len(testing_data),15, 64), 10, 10)





