import math
import pickle
import subprocess
from pathlib import Path
import sys
import time
from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from model import Model
sys.path.append(str((Path(__file__)/"../../../../../utils").resolve()))
sys.path.append(str((Path(__file__)/"../../../../../evaluation").resolve()))
from Config import Config
from metrics import Metric

def scale(data):
    for user in data:
        for session in user:
            for i in range(len(session)):
                # Keystroke scaling
                for j in range(10):
                    if (j == 9):
                        # key code
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

class TrainDataset(Dataset):
    def __init__(self, training_data, batch_size, epoch_batch_count):
        self.training_data = training_data
        self.batch_size = batch_size
        self.epoch_batch_count = epoch_batch_count

    def __len__(self):
        return self.batch_size * self.epoch_batch_count;

    def __getitem__(self, idx):
        genuine_user_idx = np.random.randint(0, len(self.training_data))
        imposter_user_idx = np.random.randint(0, len(self.training_data))
        while (imposter_user_idx == genuine_user_idx):
            imposter_user_idx = np.random.randint(0, len(self.training_data))
        
        genuine_sess_1 = np.random.randint(0, len(self.training_data[0]))
        genuine_sess_2 = np.random.randint(0, len(self.training_data[0]))
        while (genuine_sess_2 == genuine_sess_1):
            genuine_sess_2 = np.random.randint(0, len(self.training_data[0]))
        imposter_sess = np.random.randint(0, len(self.training_data[0]))
        
        genuine_seq_1 = np.random.randint(0, len(self.training_data[genuine_user_idx][genuine_sess_1]))
        genuine_seq_2 = np.random.randint(0, len(self.training_data[genuine_user_idx][genuine_sess_2]))
        imposter_seq = np.random.randint(0, len(self.training_data[imposter_user_idx][imposter_sess]))

        anchor = self.training_data[genuine_user_idx][genuine_sess_1][genuine_seq_1]
        positive = self.training_data[genuine_user_idx][genuine_sess_2][genuine_seq_2]
        negative = self.training_data[imposter_user_idx][imposter_sess][imposter_seq]

        return [anchor[0], anchor[1][:, :24]], [positive[0], positive[1][:, :24]], [negative[0], negative[1][:, :24]]

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
        
        seqs = self.eval_data[user_idx][session_idx][seq_idx]

        return [seqs[0], seqs[1][:, :24]]
    
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(dim=1).sqrt()
    
    def calc_cosine(self, x1, x2):
        dot_product_sum = (x1*x2).sum(dim=1)
        norm_multiply = (x1.pow(2).sum(dim=1).sqrt()) * (x2.pow(2).sum(dim=1).sqrt())
        return dot_product_sum / norm_multiply
    
    def calc_manhattan(self, x1, x2):
        return (x1-x2).abs().sum(dim=1)
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

def evaluate(model, testing_data, batch_size, trg_len, number_of_enrollment_sessions, number_of_verify_sessions):
    model.train(False)

    t_dataset = TestDataset(testing_data)
    t_dataloader = DataLoader(t_dataset, batch_size=batch_size)
    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(t_dataloader):
            feature_embeddings.append(model([item[0].float(), item[1].float()]))

    eer = Metric.cal_user_eer(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions, number_of_verify_sessions, "hmog")[0]
    return eer

if __name__ == "__main__":
    config=Config()
    data = config.get_config_dict()["data"]
    hyperparams = config.get_config_dict()["hyperparams"]

    train_id = config.get_config_dict()["preprocessed_data"]["hmog"]["train"]
    test_id = config.get_config_dict()["preprocessed_data"]["hmog"]["test"]
    val_id = config.get_config_dict()["preprocessed_data"]["hmog"]["val"]

    subprocess.run(f"gdown {train_id}", shell=True)
    subprocess.run(f"gdown {test_id}", shell=True)
    subprocess.run(f"gdown {val_id}", shell=True)

    if(config.get_config_dict()["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    infile = open("training_keystroke_imu_data_all.pickle",'rb')
    training_data = pickle.load(infile)
    infile.close()

    infile = open("validation_keystroke_imu_data_all.pickle",'rb')
    validation_data = pickle.load(infile)
    infile.close()

    infile = open("testing_keystroke_imu_data_all.pickle",'rb')
    testing_data = pickle.load(infile)
    infile.close()

    for user in validation_data:
        for idx, session in enumerate(user):
            user[idx] = session[:50]

    scale(validation_data)
    scale(training_data)

    batch_size = hyperparams["batch_size"]["hmog"]
    epoch_batch_count = hyperparams["epoch_batch_count"]["hmog"]
    keystroke_l = data["keystroke_sequence_len"]
    imu_l = data["imu_sequence_len"]
    keystroke_feature_count = hyperparams["keystroke_feature_count"]["hmog"]
    imu_feature_count = hyperparams["imu_feature_count"]["two_types"]
    trg_len = hyperparams["target_len"]
    number_of_enrollment_sessions = hyperparams["number_of_enrollment_sessions"]["hmog"]
    number_of_verify_sessions = hyperparams["number_of_verify_sessions"]["hmog"]

    best_model_save_path = f'{str((Path(__file__)/"../").resolve())}/best_models'
    checkpoint_save_path = f'{str((Path(__file__)/"../").resolve())}/checkpoints'

    subprocess.run(f"mkdir {best_model_save_path}", shell=True)
    subprocess.run(f"mkdir {checkpoint_save_path}", shell=True)

    dataset = TrainDataset(training_data, batch_size, epoch_batch_count)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = Model(keystroke_feature_count, imu_feature_count, keystroke_l, imu_l, trg_len)

    loss_fn = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

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
            anchor_out = model([anchor[0].float(), anchor[1].float()])
            positive_out = model([positive[0].float(), positive[1].float()])
            negative_out = model([negative[0].float(), negative[1].float()])
            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            t_loss = t_loss + loss.item()
            if batch_idx == len(dataloader)-1:
                t_loss = t_loss / len(dataloader)
    
        eer = evaluate(model, validation_data, batch_size, trg_len, number_of_enrollment_sessions, number_of_verify_sessions)
        end = time.time()
        print(f"------> Epoch No: {i+1} - Loss: {t_loss:>7f} - EER: {eer:>7f} - Time: {end-start:>2f}")
        if (eer < g_eer):
            print(f"EER improved from {g_eer} to {eer}")
            g_eer = eer
            torch.save(model, best_model_save_path + f"/epoch_{i+1}_eer_{eer}.pt")
            
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