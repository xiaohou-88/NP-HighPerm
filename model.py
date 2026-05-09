import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super().__init__()
        pe = torch.zeros(pad_size, embed)
        position = torch.arange(0, pad_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed, 2).float() * (-np.log(10000.0) / embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, pad_size, embed]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, embed]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class Scaled_Dot_Product_Attention(nn.Module):
    def forward(self, Q, K, V, scale=None, mask=None):
        attention = torch.matmul(Q, K.transpose(-2, -1))  # [batch, head, seq, seq]
        if scale:
            attention = attention * scale
        if mask is not None:
            # mask: [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super().__init__()
        self.num_head = num_head
        self.dim_head = dim_model // num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim_model = x.size()
        Q = self.fc_Q(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1,2)
        K = self.fc_K(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1,2)
        V = self.fc_V(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1,2)
        scale = self.dim_head ** -0.5
        # mask shape: [batch_size, 1, 1, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        context = self.attention(Q, K, V, scale, mask)
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, self.num_head * self.dim_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class CustomEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super().__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x, mask=None):
        out = self.attention(x, mask)
        out = self.feed_forward(out)
        return out

class CustomTransformerEncoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, num_layers, dropout, seq_len):
        super().__init__()
        self.pos_enc = Positional_Encoding(dim_model, seq_len, dropout)
        self.layers = nn.ModuleList([
            CustomEncoderLayer(dim_model, num_head, hidden, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EnhancedGCN(nn.Module):

class ConditionAwareFusion(nn.Module):

  
class MultiModalNet(nn.Module):
    

    

