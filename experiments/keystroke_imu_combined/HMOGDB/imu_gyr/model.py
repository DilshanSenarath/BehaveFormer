from torch import nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, k, d_model, seq_len):
        super().__init__()
        
        self.embedding = nn.Parameter(torch.zeros([k, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions = torch.tensor([i for i in range(seq_len)], requires_grad=False).unsqueeze(1).repeat(1, k)
        s = 0.0
        interval = seq_len / k
        mu = []
        for _ in range(k):
            mu.append(nn.Parameter(torch.tensor(s, dtype=torch.float), requires_grad=True))
            s = s + interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.tensor([torch.tensor([50.0], dtype=torch.float, requires_grad=True) for _ in range(k)]).unsqueeze(0))
        
    def normal_pdf(self, pos, mu, sigma):
        a = pos - mu
        log_p = -1*torch.mul(a, a)/(2*(sigma**2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(pdfs, self.embedding)
        
        return inputs + pos_enc.unsqueeze(0).repeat(inputs.size(0), 1, 1)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, _heads, dropout, seq_len):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self._attention = nn.MultiheadAttention(seq_len, _heads, batch_first=True)
        
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.cnn_units = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_units, (1, 1)),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, self.cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(self.cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(self.cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src = self.attn_norm(src + self.attention(src, src, src)[0] + self._attention(src.transpose(-1, -2), src.transpose(-1, -2), src.transpose(-1, -2))[0].transpose(-1, -2))
        
        src = self.final_norm(src + self.cnn(src.unsqueeze(dim=1)).squeeze(dim=1))
            
        return src
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, _heads, seq_len, num_layer=2, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(TransformerEncoderLayer(d_model, heads, _heads, dropout, seq_len))

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        return src
    
class Transformer(nn.Module):
    def __init__(self, num_layer, d_model, k, heads, _heads, seq_len, trg_len, dropout):
        super(Transformer, self).__init__()

        self.pos_encoding = PositionalEncoding(k, d_model, seq_len)

        self.encoder = nn.DataParallel(TransformerEncoder(d_model, heads, _heads, seq_len, num_layer, dropout))

    def forward(self, inputs):
        encoded_inputs = self.pos_encoding(inputs)

        return self.encoder(encoded_inputs)
    
class Model(nn.Module):
    def __init__(self, keystroke_feature_count, imu_feature_count, keystroke_l, imu_l, trg_len):
        super(Model, self).__init__()
        
        self.keystroke_transformer = Transformer(5, keystroke_feature_count, 20, 5, 10, keystroke_l, trg_len, 0.1)
        self.imu_transformer = Transformer(5, imu_feature_count, 20, 6, 10, imu_l, trg_len, 0.1)
        
        self.linear_key = nn.Sequential(
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(250, trg_len),
            nn.ReLU()
        )
        
        self.linear_imu = nn.Sequential(
            nn.Linear(1200, 600),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(600, trg_len),
            nn.ReLU()
        )
        
        self.linear_final = nn.Linear(128, trg_len)
        
    def forward(self, inputs):
        keystroke_inputs, imu_inputs = inputs
        
        keystroke_out = self.linear_key(torch.flatten(self.keystroke_transformer(keystroke_inputs), start_dim=1, end_dim=2))
        imu_out = self.linear_imu(torch.flatten(self.imu_transformer(imu_inputs), start_dim=1, end_dim=2))
        
        concat_out = torch.concat([keystroke_out, imu_out], dim=-1)
        
        return self.linear_final(concat_out)
    
