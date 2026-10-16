import io
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from feature_engineering import MolDataset, smi_tokenizer
from model import MultiModalNet


plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


class ModifiedMultiHeadAttention(nn.Module):
    """Multi-head attention wrapper that stores attention weights."""

    def __init__(self, original_attention):
        super().__init__()
        for key, value in original_attention.__dict__.items():
            setattr(self, key, value)
        self.attention_scores = None

    def forward(self, x, mask=None):
        batch_size, seq_len, dim_model = x.size()
        Q = self.fc_Q(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        K = self.fc_K(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)
        V = self.fc_V(x).view(batch_size, seq_len, self.num_head, self.dim_head).transpose(1, 2)

        
        return out


def get_atom_weights_from_attention(mol, smiles, tokens, token_attention):
    """Convert token attention weights into per-atom weights."""
    num_atoms = mol.GetNumAtoms()
    atom_weights = np.zeros(num_atoms)

    

    return atom_weights


def smooth_atom_weights(atom_weights, mol, smoothing_factor=0.3):
    """Smooth atom weights so neighboring atoms have closer values."""
    num_atoms = len(atom_weights)
    smoothed_weights = atom_weights.copy()

    

    return smoothed_weights


def create_yellow_orange_red_colormap():
    """Create a custom yellow-to-orange-to-red colormap."""
    colors = [
        '#FFEB3B',  # Light yellow, lowest weight
        '#FFD54F',  # Bright yellow
        '#FFC107',  # Standard yellow
        '#FFB300',  # Dark yellow
        '#FF9800',  # Standard orange
        '#FF6F00',  # Dark orange
        '#FF5722',  # Orange-red
        '#E64A19',  # Dark orange-red
        '#D32F2F',  # Red
        '#B71C1C',  # Dark red, highest weight
    ]

    return LinearSegmentedColormap.from_list('yellow_orange_red', colors, N=256)





def generate_mol_image(mol, atom_weights, threshold_percentile=80, colormap='yellow_orange_red', enhance_high_weight=True):
    """Generate a single RDKit molecule image with high-weight atoms highlighted."""
    threshold = np.percentile(atom_weights, threshold_percentile)
    high_weight_atoms = []
    atom_colors = {}
    atom_radii = {}
    cmap_mol = get_colormap(colormap)

   

    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def create_combined_attention_visualization(mols_info, save_path, colormap='yellow_orange_red', labels=None):
    """
    Combine three molecule attention plots into one figure with a shared colorbar.

    mols_info is a list of dictionaries containing molecule metadata and atom weights.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})
    fig.patch.set_facecolor('white')

    if labels is None:
        labels = ["Low Property", "Medium Property", "High Property"]

   

    cbar_ax = axes[3]
    cmap = get_colormap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('Relative Attention Weight', fontsize=12, fontweight='bold')
    cb.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined attention figure: {save_path}")


def create_molecule_attention_visualization(
    mol,
    smiles,
    tokens,
    attention_weights,
    save_path,
    mol_id="Unknown",
    true_perm=-7.14,
    pred_perm=-7.20,
    use_smooth=True,
    colormap='yellow_orange_red',
    enhance_high_weight=True,
    threshold_percentile=80,
):
    """
    Create a molecule attention visualization.

    Only atoms above the selected attention-weight percentile are colored.
    """
    base_output_dir = os.path.dirname(save_path)
    mol_folder = os.path.join(base_output_dir, str(mol_id))
    os.makedirs(mol_folder, exist_ok=True)

    filename = os.path.basename(save_path)
    save_path = os.path.join(mol_folder, filename)

   


def create_multiple_threshold_versions(
    mol,
    smiles,
    tokens,
    attention_weights,
    output_dir,
    mol_id="Unknown",
    true_perm=-7.14,
    pred_perm=-7.20,
):
    """Generate attention visualizations at several percentile thresholds."""
    mol_folder = os.path.join(output_dir, str(mol_id))
    os.makedirs(mol_folder, exist_ok=True)

    thresholds = {
        50: 'top 50% weighted atoms',
        60: 'top 40% weighted atoms',
        70: 'top 30% weighted atoms (recommended)',
        80: 'top 20% weighted atoms',
        85: 'top 15% weighted atoms',
        90: 'top 10% weighted atoms',
    }

    


def main():
    """Run TPSA-based molecule attention visualization."""
    import random
    import time
    import pandas as pd

    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "new_result", "01weight_best_result", "fold_5", "best_model.pth")
    test_file = os.path.join(base_dir, "all_data_split2", "folds", "fold_5_test.csv")

    output_dir = os.path.join(base_dir, "attention_visualization_results", "polarity_based")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nLoading model...")
    model = MultiModalNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for layer in model.transformer.layers:
        original_attn = layer.attention
        layer.attention = ModifiedMultiHeadAttention(original_attn)

   
   


if __name__ == "__main__":
    main()
