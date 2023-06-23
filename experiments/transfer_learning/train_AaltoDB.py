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
from torch import nn
from  model import KeystrokeTransformer
import math
import time
from metrics import Metric

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

# Dataset for training the model
class TrainDataset(Dataset):
    def __init__(self, training_data, batch_size, epoch_batch_count, seq_len):
        self.training_data = training_data
        self.batch_size = batch_size
        self.epoch_batch_count = epoch_batch_count
        self.seq_len = seq_len

    def __len__(self):
        return self.batch_size * self.epoch_batch_count;

    def __getitem__(self, idx):
        genuine_user_idx = np.random.randint(0, len(self.training_data))
        imposter_user_idx = np.random.randint(0, len(self.training_data))
        while (imposter_user_idx == genuine_user_idx):
            imposter_user_idx = np.random.randint(0, len(self.training_data))
        
        genuine_seq_1 = np.random.randint(0, 15)
        genuine_seq_2 = np.random.randint(0, 15)
        while (genuine_seq_1 == genuine_seq_2):
            genuine_seq_2 = np.random.randint(0, 15)
        imposter_seq = np.random.randint(0, 15)

        anchor = self.pad_sequence(self.training_data[genuine_user_idx][genuine_seq_1])
        positive = self.pad_sequence(self.training_data[genuine_user_idx][genuine_seq_2])
        negative = self.pad_sequence(self.training_data[imposter_user_idx][imposter_seq])

        return anchor, positive, negative

    def pad_sequence(self, sequence):
        if (len(sequence) == self.seq_len):
            return sequence
        elif (len(sequence) < self.seq_len):
            row_count = self.seq_len - len(sequence)
            return np.append(sequence, np.array([0.0] * 10 * row_count).reshape(row_count, 10), axis=0)
        else:
            return sequence[0:self.seq_len]
        
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

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(dim=1).sqrt()
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

if __name__ == "__main__":
    config = Config()
    keystroke_data_id = config.get_config_dict()["preprocessed_data"]["aalto"]["keystroke"]

    subprocess.run(f"gdown {keystroke_data_id}", shell=True)

    if (config.get_config_dict()["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    preprocess()

    # Load training data
    infile = open("training_data.pickle",'rb')
    training_data = pickle.load(infile)
    infile.close()

    # Load validation data
    infile = open("validation_data.pickle",'rb')
    validation_data = pickle.load(infile)
    infile.close()

    extract_normalize_features(training_data)
    extract_normalize_features(validation_data)

    batch_size = config.get_config_dict()["hyperparams"]["batch_size"]["aalto"]
    epoch_batch_count = config.get_config_dict()["hyperparams"]["epoch_batch_count"]["aalto"]
    l = config.get_config_dict()["data"]["keystroke_sequence_len"]
    feature_count = config.get_config_dict()["hyperparams"]["keystroke_feature_count"]["aalto"]
    trg_len = config.get_config_dict()["hyperparams"]["target_len"]

    best_model_save_path = f"{str((Path(__file__)/'../').resolve())}/best_models"
    checkpoint_save_path = f"{str((Path(__file__)/'../').resolve())}/checkpoints"

    subprocess.run(f"mkdir {best_model_save_path}", shell=True)
    subprocess.run(f"mkdir {checkpoint_save_path}", shell=True)

    dataset = TrainDataset(training_data, batch_size, epoch_batch_count, l)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    val_dataset = TestDataset(validation_data, batch_size, l)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    model = KeystrokeTransformer(6, feature_count, 20, 5, 10, l, trg_len, 0.1)

    loss_fn = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get_config_dict()["hyperparams"]["learning_rate"])

    g_eer = math.inf
    epochs = int(sys.argv[1])
    init_epoch = 0

    if (len(sys.argv) > 2):
        if (config.get_config_dict()["GPU"] == "True"):
            checkpoint = torch.load(f"{checkpoint_save_path}/training_{sys.argv[2]}.tar")
        else:
            checkpoint = torch.load(f"{checkpoint_save_path}/training_{sys.argv[2]}.tar", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        init_eer = checkpoint['eer']

        epochs = init_epoch + epochs
        g_eer = init_eer

    for i in range(init_epoch, epochs):
        t_loss = 0.0
        start = time.time()
        model.train(True)
        for batch_idx, item in enumerate(dataloader):
            anchor, positive, negative = item
            optimizer.zero_grad()
            anchor_out = model(anchor.float())
            positive_out = model(positive.float())
            negative_out = model(negative.float())
            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            t_loss = t_loss + loss.item()
            if batch_idx == len(dataloader)-1:
                t_loss = t_loss / len(dataloader)
        
        model.train(False)
        
        with torch.no_grad():
            feature_embeddings = []
            for batch_idx, item in enumerate(val_dataloader):
                feature_embeddings.append(model(item.float()))
    
        eer = Metric.cal_user_eer_aalto(torch.cat(feature_embeddings, dim=0).view(len(validation_data),15, trg_len), 10, 5)[0]
        end = time.time()
        print(f"------> Epoch No: {i+1} - Loss: {t_loss:>7f} - EER: {eer:>4f} - Time: {end-start:>2f}")
        if (eer < g_eer):
            torch.save(model, best_model_save_path + f"/epoch_{i+1}_eer_{eer}.pt")
            print(f"Model saved - EER improved from {g_eer} to {eer}")
            g_eer = eer
            
        if ((i+1) % 50 == 0):
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eer': g_eer
            }, f"{checkpoint_save_path}/training_{i+1}.tar")

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'eer': g_eer
    }, f"{checkpoint_save_path}/training_{epochs}.tar")






