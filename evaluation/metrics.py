import torch
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

plt.style.use('seaborn-v0_8-bright')
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams.update({"axes.grid" : True, "grid.color": "black"})
mpl.rc('axes',edgecolor='black')
mpl.rcParams.update({'font.size': 13})

class Metric:
    @staticmethod
    def eer_compute(scores_g, scores_i):
        far = []
        frr = []
        thresholds = []
        ini=torch.min(torch.cat([scores_g, scores_i], dim=0)).item()
        fin=torch.max(torch.cat([scores_g, scores_i], dim=0)).item()
        
        paso=(fin-ini)/10000
        threshold = ini-paso
        while threshold < fin+paso:
            far.append(torch.count_nonzero(scores_i >= threshold).item()/scores_i.shape[0])
            frr.append(torch.count_nonzero(scores_g < threshold).item()/scores_g.shape[0])
            thresholds.append(threshold)
            threshold = threshold + paso
        
        gap = torch.abs(torch.tensor(far) - torch.tensor(frr))
        j = torch.nonzero(gap == torch.min(gap))
        index = j[0][0].item()
        return ((far[index]+frr[index])/2)*100, thresholds[index]
    
    @staticmethod
    def cal_user_eer_aalto(feature_embeddings, num_enroll_sessions, num_verify_sessions):
        accs = []
        thresholds = []
        for i in range(feature_embeddings.shape[0]):
            enroll_emb = torch.unsqueeze(feature_embeddings[i,:num_enroll_sessions], dim=0)
            verify_emb = torch.unsqueeze(torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0), dim=1)

            scores = torch.mean(torch.linalg.norm(verify_emb-enroll_emb, dim=-1), dim=-1)
            acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])
            accs.append(acc)
            thresholds.append(threshold)
        
        return 100 - sum(accs)/len(accs) if len(accs) != 0 else 0, sum(thresholds)/len(thresholds) if len(thresholds) != 0 else 0

    @staticmethod
    def cal_session_distance_humi(ver_session_list, enr_session_list):
        return torch.mean(torch.linalg.norm(ver_session_list - enr_session_list.squeeze(dim=1).unsqueeze(dim=0), dim=-1), dim=-1)

    @staticmethod
    def cal_session_distance_hmog(ver_session_list, enr_session_list):
        dis = []
        for ver_sess in ver_session_list:
            temp = []
            for en_sess in enr_session_list:
                temp.append(torch.mean(torch.mean(torch.linalg.norm((ver_sess.unsqueeze(dim=1) - en_sess.unsqueeze(dim=0)), dim=-1), dim=-1), dim=-1))
            dis.append(sum(temp)/len(temp))
        return torch.tensor(dis)

    @staticmethod
    def cal_user_eer(feature_embeddings, num_enroll_sessions, num_verify_sessions, dataset):
        accs = []
        thresholds = []
        for i in range(feature_embeddings.shape[0]):
            all_ver_embeddings = torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
            if (dataset == "humi"):
                scores = Metric.cal_session_distance_humi(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            else:
                scores = Metric.cal_session_distance_hmog(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])
            accs.append(acc)
            thresholds.append(threshold)
        
        return 100 - sum(accs)/len(accs) if len(accs) != 0 else 0, sum(thresholds)/len(thresholds) if len(thresholds) != 0 else 0
    
    @staticmethod
    def calculate_usability(scores, threshold, periods, labels):
        preds = list(map(lambda x: 1 if x <= threshold else 0, scores))
        total_time = 0
        accepted_time = 0
        for i,pred in enumerate(preds):
            if (labels[i] == 0):
                continue
            total_time = total_time + periods[i]
            if (pred == 1):
                accepted_time = accepted_time + periods[i]
        return accepted_time/total_time if total_time != 0 else 0
    
    @staticmethod
    def calculate_TCR(scores, threshold, periods, labels):
        preds = list(map(lambda x: 1 if x <= threshold else 0, scores))
        values = []
        time = 0
        active = True
        for i,pred in enumerate(preds):
            if (labels[i] == 1):
                continue
            if (pred == 0 and active):
                values.append(time)
                time = 0
                active = False
            elif (pred == 1):
                time = time + periods[i]
                if (not(active)):
                    active = True
        return np.mean(values) if len(values) != 0 else 0
    
    @staticmethod
    def calculate_FRWI(scores, threshold, periods, labels):
        preds = list(map(lambda x: 1 if x <= threshold else 0, scores))
        values = []
        time = 0
        active = True
        for i,pred in enumerate(preds):
            if (labels[i] == 0):
                continue
            if (pred == 1 and active):
                values.append(time)
                time = 0
                active = False
            elif (pred == 0):
                time = time + periods[i]
                if (not(active)):
                    active = True
        return max(values) / 60 if len(values) != 0 else 0 # Convert to minutes
    
    @staticmethod
    def calculate_FAWI(scores, threshold, periods, labels):
        preds = list(map(lambda x: 1 if x <= threshold else 0, scores))
        values = []
        time = 0
        active = True
        for i,pred in enumerate(preds):
            if (labels[i] == 1):
                continue
            if (pred == 0 and active):
                values.append(time)
                time = 0
                active = False
            elif (pred == 1):
                time = time + periods[i]
                if (not(active)):
                    active = True
        return max(values) / 60 if len(values) != 0 else 0 # Convert to minutes
    
    @staticmethod
    def save_DET_curve(feature_embeddings, num_enroll_sessions, num_verify_sessions, dataset, results_path):
        min_max = [math.inf, -math.inf]
        values = 0
        
        for i in range(feature_embeddings.shape[0]):
            all_ver_embeddings = torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
            if (dataset == "humi"):
                scores = Metric.cal_session_distance_humi(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            else:
                scores = Metric.cal_session_distance_hmog(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            Metric.find_min_max_scores(scores[:num_verify_sessions], scores[num_verify_sessions:], min_max)
        
        for i in range(feature_embeddings.shape[0]):
            all_ver_embeddings = torch.cat([feature_embeddings[i,num_enroll_sessions:], torch.flatten(feature_embeddings[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(feature_embeddings[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
            if (dataset == "humi"):
                scores = Metric.cal_session_distance_humi(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            else:
                scores = Metric.cal_session_distance_hmog(all_ver_embeddings, feature_embeddings[i,:num_enroll_sessions])
            far_frrs = Metric.get_far_frr(i, scores[:num_verify_sessions], scores[num_verify_sessions:], min_max)
            if (type(values) == int):
                values = far_frrs
            else:
                values = values + far_frrs
                
        values = values / feature_embeddings.shape[0]
        
        df = pd.DataFrame(values, columns=["FAR", "FRR"])
        df.to_csv(f"{results_path}/far-frr.csv")

    @staticmethod
    def find_min_max_scores(scores_g, scores_i, min_max):
        ini=torch.min(torch.cat([scores_g, scores_i], dim=0)).item()
        fin=torch.max(torch.cat([scores_g, scores_i], dim=0)).item()
        
        if (ini < min_max[0]):
            min_max[0] = ini
        if (fin > min_max[1]):
            min_max[1] = fin

    @staticmethod
    def get_far_frr(user_id, scores_g, scores_i, min_max):
        far_frr = []
        ini = min_max[0]
        fin = min_max[1]

        paso=(fin-ini)/10000
        threshold = ini-paso
        while threshold < fin+paso:
            far = torch.count_nonzero(scores_i <= threshold).item()/scores_i.shape[0] * 100
            frr = torch.count_nonzero(scores_g > threshold).item()/scores_g.shape[0] * 100
            far_frr.append([far, frr])
            threshold = threshold + paso
            
        return np.array(far_frr)
    
    @staticmethod
    def save_PCA_curve(feature_embeddings, session_count, number_of_users, results_path):
        users = []
        for i in range(number_of_users):
            val = np.random.randint(0, len(feature_embeddings))
            while (val in users):
                val = np.random.randint(0, len(feature_embeddings))
            users.append(val)
        
        values = TSNE(n_iter=1000, perplexity=7).fit_transform(feature_embeddings[users].mean(dim=-2).flatten(start_dim=0, end_dim=1).cpu().numpy())
        
        labels = []
        for user in users:
            for i in range(session_count):
                labels.append(user)

        pd.DataFrame([[silhouette_score(values, labels)]], columns=["Silhouette Score"]).to_csv(f"{results_path}/silhouette_score.csv")
        
        df = pd.DataFrame(values, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
        labels = []
        for i in range(number_of_users):
            for j in range(session_count):
                labels.append(f"User {i+1}")
        df["Users"] = labels

        g = sns.relplot(
            data=df,
            x="t-SNE Dimension 1", y="t-SNE Dimension 2",
            hue="Users",
            sizes=(10, 200),
        )
        g.set(xscale="linear", yscale="linear")
        g.ax.xaxis.grid(True, "minor", linewidth=.25)
        g.ax.yaxis.grid(True, "minor", linewidth=.25)
        g.despine(left=True, bottom=True)
        plt.savefig(f'{results_path}/pca_graph.png', dpi=400)
