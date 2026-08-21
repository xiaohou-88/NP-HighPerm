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

        scale = self.dim_head ** -0.5
        attention = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(attention, dim=-1)
        self.attention_scores = attention

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head * self.dim_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


def get_atom_weights_from_attention(mol, smiles, tokens, token_attention):
    """Convert token attention weights into per-atom weights."""
    num_atoms = mol.GetNumAtoms()
    atom_weights = np.zeros(num_atoms)

    valid_tokens = [t for t in tokens if t != 'PAD']
    valid_attention = token_attention[:len(valid_tokens)]

    weight_per_atom = np.sum(valid_attention) / num_atoms
    atom_weights = np.full(num_atoms, weight_per_atom)

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol != 'C':
            atom_weights[i] *= 1.5
        if atom.GetIsAromatic():
            atom_weights[i] *= 1.2

    if atom_weights.sum() > 0:
        atom_weights = atom_weights / atom_weights.sum() * len(valid_attention)

    return atom_weights


def smooth_atom_weights(atom_weights, mol, smoothing_factor=0.3):
    """Smooth atom weights so neighboring atoms have closer values."""
    num_atoms = len(atom_weights)
    smoothed_weights = atom_weights.copy()

    for _ in range(2):
        new_weights = smoothed_weights.copy()
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

            if len(neighbors) > 0:
                neighbor_avg = np.mean([smoothed_weights[n] for n in neighbors])
                new_weights[i] = (1 - smoothing_factor) * smoothed_weights[i] + smoothing_factor * neighbor_avg

        smoothed_weights = new_weights

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


def get_colormap(colormap_name):
    """Get a colormap while remaining compatible with old matplotlib versions."""
    if colormap_name == 'yellow_orange_red':
        return create_yellow_orange_red_colormap()

    try:
        return colormaps.get_cmap(colormap_name)
    except (AttributeError, KeyError):
        return cm.get_cmap(colormap_name)


def generate_mol_image(mol, atom_weights, threshold_percentile=80, colormap='yellow_orange_red', enhance_high_weight=True):
    """Generate a single RDKit molecule image with high-weight atoms highlighted."""
    threshold = np.percentile(atom_weights, threshold_percentile)
    high_weight_atoms = []
    atom_colors = {}
    atom_radii = {}
    cmap_mol = get_colormap(colormap)

    high_weight_indices = [i for i, weight in enumerate(atom_weights) if weight > threshold]
    if len(high_weight_indices) > 0:
        high_weights = atom_weights[high_weight_indices]
        weight_min = high_weights.min()
        weight_max = high_weights.max()
        norm = Normalize(vmin=weight_min, vmax=weight_max)

        for i in high_weight_indices:
            high_weight_atoms.append(i)
            color_val = norm(atom_weights[i])
            color_val = np.clip(color_val, 0, 1)
            rgb = cmap_mol(color_val)[:3]
            atom_colors[i] = rgb

            if enhance_high_weight:
                base_radius = 0.4
                if color_val > 0.7:
                    radius = base_radius + (color_val ** 1.5) * 0.5
                else:
                    radius = base_radius + color_val * 0.3
            else:
                radius = 0.5 + (color_val * 0.3)
            atom_radii[i] = float(radius)

    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().highlightRadius = 0.6
    drawer.drawOptions().setHighlightColour((1.0, 1.0, 1.0))
    drawer.drawOptions().bondLineWidth = 2.5
    drawer.drawOptions().atomLabelFontSize = 18

    atom_colors_converted = {k: tuple(map(float, v)) for k, v in atom_colors.items()}

    if len(high_weight_atoms) > 0:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=high_weight_atoms,
            highlightAtomColors=atom_colors_converted,
            highlightAtomRadii=atom_radii,
        )
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

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

    for i, (info, ax) in enumerate(zip(mols_info, axes[:3])):
        ax.axis('off')

        mol_img = generate_mol_image(
            info['mol'],
            info['atom_weights'],
            threshold_percentile=80,
            colormap=colormap,
        )
        ax.imshow(mol_img)

        title_text = f"{labels[i]}\n"
        title_text += f"ID: {info['mol_id']}\n"
        if 'tpsa' in info:
            title_text += f"TPSA: {info['tpsa']:.2f}\n"
        elif 'logp' in info:
            title_text += f"LogP: {info['logp']:.2f}\n"
        title_text += f"True: {info['true_perm']:.2f} | Pred: {info['pred_perm']:.2f}"
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=10)

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

    if len(attention_weights.shape) == 3:
        attention_weights = attention_weights.mean(axis=0)

    valid_len = len(tokens)
    for i, token in enumerate(tokens):
        if token == 'PAD':
            valid_len = i
            break

    tokens = tokens[:valid_len]
    attention_weights = attention_weights[:valid_len, :valid_len]
    token_attention = attention_weights.mean(axis=0)

    atom_weights = get_atom_weights_from_attention(mol, smiles, tokens, token_attention)

    if use_smooth:
        atom_weights = smooth_atom_weights(atom_weights, mol, smoothing_factor=0.4)

    if enhance_high_weight:
        atom_weights_normalized = (atom_weights - atom_weights.min()) / (atom_weights.max() - atom_weights.min() + 1e-8)
        atom_weights_enhanced = np.power(atom_weights_normalized, 0.7)
        atom_weights = atom_weights.min() + atom_weights_enhanced * (atom_weights.max() - atom_weights.min())

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    threshold = np.percentile(atom_weights, threshold_percentile)
    high_weight_atoms = []
    atom_colors = {}
    atom_radii = {}

    cmap_mol = get_colormap(colormap)
    high_weight_indices = [i for i, weight in enumerate(atom_weights) if weight > threshold]

    if len(high_weight_indices) > 0:
        high_weights = atom_weights[high_weight_indices]
        weight_min = high_weights.min()
        weight_max = high_weights.max()
        norm = Normalize(vmin=weight_min, vmax=weight_max)

        for i in high_weight_indices:
            high_weight_atoms.append(i)
            color_val = norm(atom_weights[i])
            color_val = np.clip(color_val, 0, 1)
            rgb = cmap_mol(color_val)[:3]
            atom_colors[i] = rgb

            if enhance_high_weight:
                base_radius = 0.4
                if color_val > 0.7:
                    radius = base_radius + (color_val ** 1.5) * 0.5  # Range: 0.4-0.9
                else:
                    radius = base_radius + color_val * 0.3  # Range: 0.4-0.7
            else:
                base_radius = 0.5
                radius = base_radius + (color_val * 0.3)  # Range: 0.5-0.8

            atom_radii[i] = float(radius)

    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().highlightRadius = 0.6
    drawer.drawOptions().setHighlightColour((1.0, 1.0, 1.0))
    drawer.drawOptions().bondLineWidth = 2.5
    drawer.drawOptions().atomLabelFontSize = 18

    atom_colors_converted = {k: tuple(map(float, v)) for k, v in atom_colors.items()}

    if len(high_weight_atoms) > 0:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=high_weight_atoms,
            highlightAtomColors=atom_colors_converted,
            highlightAtomRadii=atom_radii,
        )
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    ax.imshow(mol_img)

    title_text = f"Molecule ID: {mol_id}\n"
    title_text += f"True Permeability: {true_perm:.2f}\n"
    title_text += f"Predicted Permeability: {pred_perm:.2f}\n"
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=5, loc='center')

    plt.tight_layout(pad=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved molecule attention visualization: {save_path}")
    print(f"  Weight range: [{atom_weights.min():.3f}, {atom_weights.max():.3f}]")
    print(f"  Threshold ({threshold_percentile}%): {threshold:.3f}")
    print(f"  High-weight atom count: {len(high_weight_atoms)} / {len(atom_weights)}")

    if high_weight_atoms:
        print("  High-weight atom details:")
        high_weight_info = [(i, atom_weights[i]) for i in high_weight_atoms]
        high_weight_info.sort(key=lambda x: x[1], reverse=True)

        for atom_idx, weight in high_weight_info:
            atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            print(f"    Atom {atom_idx} ({atom_symbol}): {weight:.3f}")
    else:
        print("  No atoms exceeded the attention-weight threshold.")


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

    print("\nGenerating multiple threshold versions...")
    for threshold, description in thresholds.items():
        save_path = os.path.join(mol_folder, f'{mol_id}_attention_threshold_{threshold}.png')
        print(f"  Generating threshold {threshold}% ({description})...")
        create_molecule_attention_visualization(
            mol=mol,
            smiles=smiles,
            tokens=tokens,
            attention_weights=attention_weights,
            save_path=save_path,
            mol_id=mol_id,
            true_perm=true_perm,
            pred_perm=pred_perm,
            use_smooth=True,
            colormap='yellow_orange_red',
            enhance_high_weight=True,
            threshold_percentile=threshold,
        )


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

    test_df = pd.read_csv(test_file)
    dataset = MolDataset(test_file)

    print("Calculating molecular TPSA values...")
    tpsa_list = []
    for smiles in test_df['Standardise_SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            tpsa_list.append(Descriptors.TPSA(mol))
        else:
            tpsa_list.append(0)
    test_df['TPSA'] = tpsa_list

    tpsa_25 = test_df['TPSA'].quantile(0.25)
    tpsa_75 = test_df['TPSA'].quantile(0.75)

    low_candidates = test_df[test_df['TPSA'] < tpsa_25].index.tolist()
    medium_candidates = test_df[(test_df['TPSA'] >= tpsa_25) & (test_df['TPSA'] <= tpsa_75)].index.tolist()
    high_candidates = test_df[test_df['TPSA'] > tpsa_75].index.tolist()

    selected_indices = {
        'low': random.choice(low_candidates) if low_candidates else None,
        'medium': random.choice(medium_candidates) if medium_candidates else None,
        'high': random.choice(high_candidates) if high_candidates else None,
    }

    mols_info_list = []

    for category, idx in [
        ('low', selected_indices['low']),
        ('medium', selected_indices['medium']),
        ('high', selected_indices['high']),
    ]:
        if idx is None:
            print(f"Warning: no suitable {category} sample was found.")
            continue

        sample = test_df.iloc[idx]
        smiles = sample['Standardise_SMILES']
        data = dataset[idx]
        fp = data[0].unsqueeze(0).to(device)
        graph = data[1].to(device)
        seq = data[2].unsqueeze(0).to(device)
        method_tensor = torch.tensor([1 if sample['method'] == 1 else 0]).to(device)

        with torch.no_grad():
            pred_perm = model(fp, graph, seq, method_tensor)[0, 0].item()

        last_layer = model.transformer.layers[-1]
        attention_weights = last_layer.attention.attention_scores[0].cpu().numpy()

        tokens = smi_tokenizer(smiles)
        tokens = (tokens + ["PAD"] * 128)[:128]

        valid_len = next((i for i, t in enumerate(tokens) if t == 'PAD'), len(tokens))
        token_attention = attention_weights.mean(axis=0)[:valid_len, :valid_len].mean(axis=0)

        mol = Chem.MolFromSmiles(smiles)
        atom_weights = get_atom_weights_from_attention(mol, smiles, tokens[:valid_len], token_attention)
        atom_weights = smooth_atom_weights(atom_weights, mol, smoothing_factor=0.4)

        mols_info_list.append({
            'mol': mol,
            'mol_id': sample['ID'],
            'true_perm': sample['Standardized_Value'],
            'pred_perm': pred_perm,
            'tpsa': sample['TPSA'],
            'atom_weights': atom_weights,
        })

    if len(mols_info_list) == 3:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_save_path = os.path.join(output_dir, f"combined_polarity_{timestamp}.png")

        plot_labels = [
            "Low Polarity\n(Low TPSA)",
            "Medium Polarity\n(Medium TPSA)",
            "High Polarity\n(High TPSA)",
        ]
        create_combined_attention_visualization(mols_info_list, combined_save_path, labels=plot_labels)


if __name__ == "__main__":
    main()
