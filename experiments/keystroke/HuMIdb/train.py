import subprocess
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torch import nn
import time
import math
sys.path.append(str((Path(__file__)/"../../../../utils").resolve()))
sys.path.append(str((Path(__file__)/"../../../../evaluation").resolve()))
from Config import Config
from model import Model
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
                        
                # IMU scaling
                for j in range(36):
                    if (j in [0,1,2]):
                        session[i][1][:, j] = session[i][1][:, j] / 10
                    elif (j in [3,4,5,24,25,26,27,28,29]):
                        session[i][1][:, j] = session[i][1][:, j] / 1000
                    elif (j in [15,16,17]):
                        session[i][1][:, j] = session[i][1][:, j] / 100

# Dataset for training the model
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

        anchor = self.training_data[genuine_user_idx][genuine_sess_1][genuine_seq_1][0][:, [1,2,3]]
        positive = self.training_data[genuine_user_idx][genuine_sess_2][genuine_seq_2][0][:, [1,2,3]]
        negative = self.training_data[imposter_user_idx][imposter_sess][imposter_seq][0][:, [1,2,3]]

        return anchor, positive, negative

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

        return self.eval_data[user_idx][session_idx][seq_idx][0][:, [1,2,3]]


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

def evaluate(model, testing_data, batch_size, trg_len, number_of_enrollment_sessions, num_verify_sessions):
    model.train(False)

    t_dataset = TestDataset(testing_data)
    t_dataloader = DataLoader(t_dataset, batch_size=batch_size)
    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(t_dataloader):
            feature_embeddings.append(model(item.float()))

    eer = Metric.cal_user_eer(torch.cat(feature_embeddings, dim=0).view(len(testing_data), len(testing_data[0]), len(testing_data[0][0]), trg_len), number_of_enrollment_sessions, num_verify_sessions, "humi")[0]
    return eer

if __name__ == "__main__":
    config_data = Config().get_config_dict()
    preprocessed_data = config_data['preprocessed_data']
    
    subprocess.run(f'gdown {preprocessed_data["humi"]["train"]}', shell=True)
    subprocess.run(f'gdown {preprocessed_data["humi"]["val"]}', shell=True)

    if(config_data["GPU"] == "True"):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Load dataset
    infile = open("training_keystroke_imu_data_all.pickle",'rb')
    training_data = pickle.load(infile)
    infile.close()

    infile = open("validation_keystroke_imu_data_all.pickle",'rb')
    validation_data = pickle.load(infile)
    infile.close()

    convert_to_float(training_data)
    convert_to_float(validation_data)

    for user in validation_data:
        for idx, session in enumerate(user):
            user[idx] = session[:1]

    scale(training_data)
    scale(validation_data)

    hyperparams = config_data['hyperparams']

    batch_size = hyperparams['batch_size']['humi']
    epoch_batch_count = hyperparams['epoch_batch_count']['humi']
    l = config_data["data"]["keystroke_sequence_len"]
    feature_count = hyperparams['keystroke_feature_count']['humi']
    trg_len = hyperparams['target_len']
    number_of_enrollment_sessions = hyperparams['number_of_enrollment_sessions']['humi']
    num_verify_sessions = hyperparams['number_of_verify_sessions']['humi']

    best_model_save_path = str((Path(__file__)/"../").resolve()) + "/best_models"
    checkpoint_save_path = str((Path(__file__)/"../").resolve()) + "/checkpoints"

    subprocess.run(f'mkdir {best_model_save_path}', shell=True)
    subprocess.run(f'mkdir {checkpoint_save_path}', shell=True)

    dataset = TrainDataset(training_data, batch_size, epoch_batch_count)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = Model(feature_count, l, trg_len)

    loss_fn = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    g_eer = math.inf
    epochs = int(sys.argv[1])
    init_epoch = 0

    if (len(sys.argv) > 2):
        if (config_data["GPU"] == "True"):
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
    
        eer = evaluate(model, validation_data, batch_size, trg_len, number_of_enrollment_sessions, num_verify_sessions)
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