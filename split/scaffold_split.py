import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats


warnings.filterwarnings('ignore')

# ============================================================
# 0. Configuration
# ============================================================
INPUT_CSV = 'processed_permeability_data_clipped_modified.csv'
OUTPUT_DIR = 'scaffold_5fold'
N_FOLDS = 5
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv(INPUT_CSV)
df = df.reset_index(drop=True)
print(f"Total samples: {len(df)}")
print(f"Assay distribution:\n{df['Assay'].value_counts()}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Standardized_Value range: {df['Standardized_Value'].min():.3f} ~ {df['Standardized_Value'].max():.3f}")

# ============================================================
# 2. Calculate Murcko scaffold
# ============================================================
def get_murcko_scaffold(smiles: str) -> str:
    """Calculate the Murcko scaffold and return the original SMILES on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False)
    except Exception:
        return smiles


print("\nCalculating Murcko scaffolds...")
df['scaffold'] = df['Standardise_SMILES'].apply(get_murcko_scaffold)

scaffold_counts = df['scaffold'].value_counts()
print(f"Unique scaffold count: {len(scaffold_counts)}")
print(f"Scaffold size distribution (top 10):\n{scaffold_counts.head(10)}")

singleton_count = (scaffold_counts == 1).sum()
print(f"Single-molecule scaffold count: {singleton_count} ({singleton_count / len(scaffold_counts) * 100:.1f}%)")

# ============================================================
# 3. Group scaffolds and sort by value distribution
# ============================================================
scaffold_groups = df.groupby('scaffold').agg(
    mol_count=('Standardized_Value', 'count'),
    mean_value=('Standardized_Value', 'mean'),
    label_mean=('label', 'mean'),
).reset_index().sort_values('mean_value').reset_index(drop=True)

print("\nScaffold group statistics:")
print(scaffold_groups.describe())

# ============================================================
# 4. Stratify scaffolds into 5 folds
# ============================================================
def snake_assign_folds(scaffold_groups: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """
    Assign scaffolds with a zigzag strategy.

    Scaffolds are sorted by mean_value, shuffled within equal-frequency bins,
    and assigned in alternating fold order to balance value distributions.
    """
    rng = np.random.default_rng(seed)
    scaffold_groups = scaffold_groups.copy()
    scaffold_groups['fold'] = -1

    n = len(scaffold_groups)

    n_bins = min(20, n // n_folds)
    scaffold_groups['bin'] = pd.qcut(
        scaffold_groups['mean_value'],
        q=n_bins,
        labels=False,
        duplicates='drop',
    )

    fold_counter = 0
    direction = 1

    for bin_idx in sorted(scaffold_groups['bin'].unique()):
        bin_mask = scaffold_groups['bin'] == bin_idx
        bin_indices = scaffold_groups[bin_mask].index.tolist()

        rng.shuffle(bin_indices)

        fold_seq = list(range(n_folds))[::direction]
        for i, idx in enumerate(bin_indices):
            scaffold_groups.loc[idx, 'fold'] = fold_seq[fold_counter % n_folds]
            fold_counter += 1

        direction *= -1

    scaffold_groups.drop(columns=['bin'], inplace=True)
    return scaffold_groups


scaffold_groups = snake_assign_folds(scaffold_groups, N_FOLDS, RANDOM_SEED)

scaffold_fold_map = scaffold_groups.set_index('scaffold')['fold'].to_dict()
df['fold'] = df['scaffold'].map(scaffold_fold_map)

unassigned = df['fold'].isna().sum()
if unassigned > 0:
    print(f"Warning: {unassigned} samples were not assigned to a fold; assigning them to fold 0.")
    df['fold'] = df['fold'].fillna(0).astype(int)
else:
    df['fold'] = df['fold'].astype(int)

# ============================================================
# 5. Validate distribution and save statistics
# ============================================================
print("\n" + "=" * 60)
print("Fold distribution statistics")
print("=" * 60)

fold_stats = []
for fold in range(N_FOLDS):
    fold_data = df[df['fold'] == fold]
    n_scaffolds = fold_data['scaffold'].nunique()
    stats_row = {
        'fold': fold,
        'n_samples': len(fold_data),
        'n_scaffolds': n_scaffolds,
        'mean_value': fold_data['Standardized_Value'].mean(),
        'std_value': fold_data['Standardized_Value'].std(),
        'label_1_ratio': fold_data['label'].mean(),
        'pampa_ratio': (fold_data['Assay'] == 'PAMPA').mean(),
        'caco2_ratio': (fold_data['Assay'] == 'Caco-2').mean(),
    }
    fold_stats.append(stats_row)
    print(f"Fold {fold}: {stats_row['n_samples']} samples | "
          f"{n_scaffolds} scaffolds | "
          f"mean={stats_row['mean_value']:.3f} | "
          f"std={stats_row['std_value']:.3f} | "
          f"label1={stats_row['label_1_ratio']:.3f} | "
          f"PAMPA={stats_row['pampa_ratio']:.3f}")

fold_stats_df = pd.DataFrame(fold_stats)
fold_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'fold_distribution_stats.csv'), index=False)

# ============================================================
# 6. KS test for fold-value distribution consistency
# ============================================================
print("\n--- KS test: each fold vs all samples; lower statistic is better ---")
all_values = df['Standardized_Value'].values
for fold in range(N_FOLDS):
    fold_values = df[df['fold'] == fold]['Standardized_Value'].values
    ks_stat, ks_pval = stats.ks_2samp(all_values, fold_values)
    print(f"Fold {fold}: KS statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

# ============================================================
# 7. Save 5-fold train/test CSV files
# ============================================================
print("\nSaving 5-fold datasets...")
save_cols = ['ID', 'Standardise_SMILES', 'Standardized_Value',
             'Standardized_Endpoint', 'Assay', 'method', 'label']

for fold in range(N_FOLDS):
    test_data = df[df['fold'] == fold][save_cols].reset_index(drop=True)
    train_data = df[df['fold'] != fold][save_cols].reset_index(drop=True)

    train_file = os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_train.csv')
    test_file = os.path.join(OUTPUT_DIR, f'fold_{fold + 1}_test.csv')

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Fold {fold + 1}: train={len(train_data)} | test={len(test_data)} | "
          f"train_scaffolds={train_data['Standardise_SMILES'].nunique()} | "
          f"test_scaffolds={test_data['Standardise_SMILES'].nunique()}")

# ============================================================
# 8. Visualization
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Murcko Scaffold 5-Fold Split Distribution', fontsize=14, fontweight='bold')

colors = plt.cm.Set2(np.linspace(0, 1, N_FOLDS))

ax = axes[0, 0]
folds_label = [f'Fold {i}' for i in range(N_FOLDS)]
ax.bar(folds_label, fold_stats_df['n_samples'], color=colors)
ax.set_title('Samples per Fold')
ax.set_ylabel('Sample Count')
for i, v in enumerate(fold_stats_df['n_samples']):
    ax.text(i, v + 5, str(v), ha='center', fontsize=9)

ax = axes[0, 1]
ax.bar(folds_label, fold_stats_df['mean_value'],
       yerr=fold_stats_df['std_value'], color=colors, capsize=5)
ax.set_title('Mean Permeability per Fold +/- std')
ax.set_ylabel('Standardized_Value')

ax = axes[0, 2]
ax.bar(folds_label, fold_stats_df['label_1_ratio'], color=colors)
ax.set_title('Label=1 Ratio per Fold')
ax.set_ylabel('Ratio')
ax.set_ylim(0, 1)
for i, v in enumerate(fold_stats_df['label_1_ratio']):
    ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

ax = axes[1, 0]
for fold in range(N_FOLDS):
    fold_values = df[df['fold'] == fold]['Standardized_Value']
    fold_values.plot.kde(ax=ax, label=f'Fold {fold}', color=colors[fold])
ax.set_title('Permeability KDE per Fold')
ax.set_xlabel('Standardized_Value')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.bar(folds_label, fold_stats_df['n_scaffolds'], color=colors)
ax.set_title('Scaffold Count per Fold')
ax.set_ylabel('Scaffold Count')
for i, v in enumerate(fold_stats_df['n_scaffolds']):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=9)

ax = axes[1, 2]
x = np.arange(N_FOLDS)
w = 0.35
ax.bar(x - w / 2, fold_stats_df['pampa_ratio'], width=w, label='PAMPA', color='steelblue')
ax.bar(x + w / 2, fold_stats_df['caco2_ratio'], width=w, label='Caco-2', color='coral')
ax.set_title('Assay Ratio per Fold')
ax.set_ylabel('Ratio')
ax.set_xticks(x)
ax.set_xticklabels(folds_label)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fold_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nVisualization figure saved to: {OUTPUT_DIR}/fold_distribution.png")

# ============================================================
# 9. Save scaffold assignment information
# ============================================================
scaffold_info = df[['ID', 'scaffold', 'fold']].copy()
scaffold_info.to_csv(os.path.join(OUTPUT_DIR, 'scaffold_fold_assignment.csv'), index=False)

print(f"\nAll files saved to: {OUTPUT_DIR}")
print("\nGenerated file list:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    print(f"  {f}  ({os.path.getsize(fpath) // 1024} KB)")
