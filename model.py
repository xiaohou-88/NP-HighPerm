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
        Q = self.fc_Q(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        K = self.fc_K(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        V = self.fc_V(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        scale = self.dim_head ** -0.5
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        context = self.attention(Q, K, V, scale, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head * self.dim_head)
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
    def __init__(self, in_dim, hidden_dim, out_dim, alpha=0.8, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gcn2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ConditionAwareFusion(nn.Module):
    def __init__(self, fp_dim=128, seq_dim=128, gcn_dim=64, method_dim=8):
        super().__init__()
        self.method_embedding = nn.Embedding(2, method_dim)

        self.condition_net = nn.Sequential(
            nn.Linear(method_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Feature-channel weights
        )

        self.fp_transform = nn.Linear(fp_dim, fp_dim)
        self.seq_transform = nn.Linear(seq_dim, seq_dim)
        self.gcn_transform = nn.Linear(gcn_dim, gcn_dim)

        self.output_dim = fp_dim + seq_dim + gcn_dim

    def forward(self, fp_feat, seq_feat, gcn_feat, method):
        method_emb = self.method_embedding(method)
        weights = F.softmax(self.condition_net(method_emb), dim=1)

        fp_transformed = self.fp_transform(fp_feat)
        seq_transformed = self.seq_transform(seq_feat)
        gcn_transformed = self.gcn_transform(gcn_feat)

        fp_weighted = fp_transformed * weights[:, 0:1]
        seq_weighted = seq_transformed * weights[:, 1:2]
        gcn_weighted = gcn_transformed * weights[:, 2:3]

        fused = torch.cat([fp_weighted, seq_weighted, gcn_weighted], dim=1)
        return fused


class MultiModalNet(nn.Module):
    def __init__(self, fp_dim=2215, seq_vocab_size=256, seq_emb_dim=128, seq_len=128, transformer_layers=2,
                 nhead=4, dropout=0.1, gcn_in_dim=5, gcn_hidden=64, gcn_out=64):
        super().__init__()
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.seq_emb = nn.Embedding(seq_vocab_size, seq_emb_dim, padding_idx=0)
        self.transformer = CustomTransformerEncoder(
            dim_model=seq_emb_dim,
            num_head=nhead,
            hidden=seq_emb_dim * 4,
            num_layers=transformer_layers,
            dropout=dropout,
            seq_len=seq_len
        )
        self.seq_pool = nn.AdaptiveAvgPool1d(1)

        self.gcn = EnhancedGCN(
            in_dim=gcn_in_dim,
            hidden_dim=gcn_hidden,
            out_dim=gcn_out,
            dropout=0.3
        )

        self.condition_fusion = ConditionAwareFusion(
            fp_dim=128, seq_dim=128, gcn_dim=64, method_dim=8
        )

        self.feat_extractor = nn.Sequential(
            nn.Linear(320, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

        self.method_embedding = nn.Embedding(2, 8)
        self.final_out = nn.Linear(128 + 8, 2)

    def forward(self, fp, graph, seq, method):
        fp_feat = self.fp_mlp(fp)  # [batch, 128]
        seq_mask = seq.ne(0)  # [batch, seq_len], PAD=0
        seq_emb = self.seq_emb(seq)  # [batch, seq_len, emb_dim]
        seq_feat = self.transformer(seq_emb, mask=seq_mask)  # [batch, seq_len, emb_dim]
        seq_mask = seq_mask.unsqueeze(-1).type_as(seq_feat)
        seq_feat = (seq_feat * seq_mask).sum(dim=1) / seq_mask.sum(dim=1).clamp(min=1.0)  # [batch, emb_dim]

        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.gcn(x, edge_index)
        gcn_feat = global_mean_pool(x, batch)

        feat = self.condition_fusion(fp_feat, seq_feat, gcn_feat, method)
        feat = self.feat_extractor(feat)  # [batch, 128]

        method_emb = self.method_embedding(method)  # [batch, 8]
        combined_feat = torch.cat([feat, method_emb], dim=1)  # [batch, 136]

        out = self.final_out(combined_feat)
        return out
