import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint, MACCSkeys
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch_geometric.data import Data, Batch

def collate_fn(batch):
    smiles_fps, graphs, fastas, reg_labels, cls_labels, methods = zip(*batch)
    batched_smiles = torch.stack(smiles_fps)
    batched_graphs = Batch.from_data_list(graphs)
    batched_fastas = torch.stack(fastas)
    batched_reg_labels = torch.stack(reg_labels)
    batched_cls_labels = torch.stack(cls_labels)
    batched_methods = torch.stack(methods)
    
    return batched_smiles, batched_graphs, batched_fastas, batched_reg_labels, batched_cls_labels, batched_methods

def atom_features(atom):
    symbol = atom.GetSymbol()
    degree = atom.GetDegree()
    is_aromatic = atom.GetIsAromatic()
    is_hydrogen = (symbol == "H")
    is_in_ring = atom.IsInRing()
    return [
        atom.GetAtomicNum(),           
        degree,                        
        int(is_aromatic),              
        int(is_hydrogen),              
        int(is_in_ring),               
    ]
 
def bond_features(bond):

    bond_type = int(bond.GetBondTypeAsDouble())  
    is_aromatic = bond.GetIsAromatic()
    is_in_ring = bond.IsInRing()
    return [
        bond_type,
        int(is_aromatic),
        int(is_in_ring),
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((1,5), dtype=np.float32), np.zeros((2,1), dtype=np.int64), np.zeros((1,3), dtype=np.float32)
    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    
    edge_index = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  
        bf = bond_features(bond)
        edge_feats.append(bf)
        edge_feats.append(bf)
    node_feats = np.array(node_feats, dtype=np.float32)
    edge_index = np.array(edge_index, dtype=np.int64).T  # shape: [2, num_edges]
    edge_feats = np.array(edge_feats, dtype=np.float32)
    return node_feats, edge_index, edge_feats

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule string.
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    return tokens

def build_vocab(smiles_list, pad_token="PAD"):
    vocab = set()
    for smi in smiles_list:
        vocab.update(smi_tokenizer(smi))
    vocab = [pad_token] + sorted(list(vocab))
    word2id = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2id

class MolDataset(Dataset):
    def __init__(self, csv_file, max_len=128):
        df = pd.read_csv(csv_file)
        self.smiles = df['Standardise_SMILES'].tolist()
        self.labels = df['Standardized_Value'].values.astype(np.float32)
        self.cls_labels = df['label'].values.astype(np.int64)
        self.methods = df['method'].values.astype(np.int64)

        # morgan fingerprint
        self.morgan_generator = GetMorganGenerator(radius=2, fpSize=2048)

        # self.fps = [self.smiles_to_fp(s) for s in self.smiles]
        self.fps = [self.generate_combined_fp(s) for s in self.smiles]
        # 分词编码
        self.max_len = max_len
        self.vocab, self.word2id = build_vocab(self.smiles)
        self.encoded = [self.encode(smi) for smi in self.smiles]

    def smiles_to_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048, dtype=np.float32)
        fp = self.generator.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return arr
    
    def generate_combined_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048 + 2048, dtype=np.float32)
                
        morgan_fp = self.morgan_generator.GetFingerprint(mol)
        morgan_arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_arr)
            
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)
        
        combined_fp = np.concatenate([morgan_arr, maccs_arr])
        return combined_fp

    def encode(self, smi):
        tokens = smi_tokenizer(smi)
        if len(tokens) < self.max_len:
            tokens += ["PAD"] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        ids = [self.word2id[token] for token in tokens]
        return np.array(ids, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fp = torch.tensor(self.fps[idx])
        seq = torch.tensor(self.encoded[idx])

        reg_label = torch.tensor(self.labels[idx])
        cls_label = torch.tensor(self.cls_labels[idx])

        method = torch.tensor(self.methods[idx], dtype=torch.long)

        node_feats, edge_index, _ = smiles_to_graph(self.smiles[idx])
        node_feats = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        graph = Data(x=node_feats, edge_index=edge_index)
        return fp, graph, seq, reg_label, cls_label, method