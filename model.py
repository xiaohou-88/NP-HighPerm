import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool



class EnhancedGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, alpha=0.8, dropout=0.3):
        super().__init__()
      

class ConditionAwareFusion(nn.Module):
    def __init__(self, fp_dim=128, seq_dim=128, gcn_dim=64, method_dim=8):
        super().__init__()
        


class MultiModalNet(nn.Module):
    def __init__(self, fp_dim=2215, seq_vocab_size=256, seq_emb_dim=128, seq_len=128, transformer_layers=2,
                 nhead=4, dropout=0.1, gcn_in_dim=5, gcn_hidden=64, gcn_out=64):
        super().__init__()
      

    def forward(self, fp, graph, seq, method):
        
        return out
