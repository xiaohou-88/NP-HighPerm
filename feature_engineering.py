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



def atom_features(atom):


def bond_features(bond):


def smiles_to_graph(smiles):


def smi_tokenizer(smi):



def build_vocab(smiles_list, pad_token="PAD", unk_token=None):



class MolDataset(Dataset):
    def __init__(self, csv_file, max_len=128, vocab=None, word2id=None, unk_token="UNK"):

    def smiles_to_fp(self, smiles):


    def generate_combined_fp(self, smiles):


    def encode(self, smi):


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

