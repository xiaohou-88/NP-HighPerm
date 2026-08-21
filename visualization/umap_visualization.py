import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from feature_engineering import MolDataset, collate_fn
from model import MultiModalNet


warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


class FeatureExtractor:
    """Extract feature representations for each modality from a trained model."""

    def __init__(self, model_path, device):
        self.device = device
        self.model = MultiModalNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def extract_features(self, data_loader):
        """
        Extract all modality features.

        Returns:
            features_dict: Dictionary containing fp, seq, graph, and fused features.
            labels_reg: Regression labels.
            labels_cls: Classification labels.
        """
        fp_features = []
        seq_features = []
        graph_features = []
        fused_features = []
        labels_reg = []
        labels_cls = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                fps, graphs, seqs, targets_reg, targets_cls, methods = batch

                fps = fps.to(self.device)
                seqs = seqs.to(self.device)
                graphs = graphs.to(self.device)
                methods = methods.to(self.device)

                fp_feat = self.model.fp_mlp(fps)
                fp_features.append(fp_feat.cpu().numpy())

                seq_emb = self.model.seq_emb(seqs)
                mask = (seqs != 0).int()
                seq_trans = self.model.transformer(seq_emb, mask=mask)
                seq_feat = self.model.seq_pool(seq_trans.transpose(1, 2)).squeeze(-1)
                seq_features.append(seq_feat.cpu().numpy())

                x, edge_index, batch_idx = graphs.x, graphs.edge_index, graphs.batch
                x = self.model.gcn(x, edge_index)
                from torch_geometric.nn import global_mean_pool
                gcn_feat = global_mean_pool(x, batch_idx)
                graph_features.append(gcn_feat.cpu().numpy())

                fused = self.model.condition_fusion(fp_feat, seq_feat, gcn_feat, methods)
                fused = self.model.feat_extractor(fused)
                fused_features.append(fused.cpu().numpy())

                labels_reg.append(targets_reg.cpu().numpy())
                labels_cls.append(targets_cls.cpu().numpy())

        features_dict = {
            'fp': np.vstack(fp_features),
            'seq': np.vstack(seq_features),
            'graph': np.vstack(graph_features),
            'fused': np.vstack(fused_features),
        }

        labels_reg = np.concatenate(labels_reg)
        labels_cls = np.concatenate(labels_cls)

        return features_dict, labels_reg, labels_cls


def categorize_permeability(values, method='quantile'):
    """
    Group permeability values into low, mid, and high categories.

    Args:
        values: Array of permeability values.
        method: Categorization method, either quantile or threshold.

    Returns:
        categories: Category labels, where 0=low, 1=mid, and 2=high.
        category_names: Category name list.
    """
    if method == 'quantile':
        q33 = np.percentile(values, 33.33)
        q66 = np.percentile(values, 66.67)
        categories = np.zeros_like(values, dtype=int)
        categories[values > q33] = 1
        categories[values > q66] = 2
    else:
        categories = np.zeros_like(values, dtype=int)
        categories[values > -6.5] = 1
        categories[values > -5.5] = 2

    category_names = ['Low', 'Mid', 'High']
    return categories, category_names


def plot_umap_single(embeddings, labels, title, save_path, category_names):
    """Plot a single UMAP embedding."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = ["#F1CF81", "#8DB4E3", "#F1918F"]

    for i, (cat_name, color) in enumerate(zip(category_names, colors)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=color,
                label=cat_name,
                alpha=0.6,
                s=30,
            )

    ax.set_xlabel('UMAP 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.tick_params(labelsize=11, width=1.5, length=5)
    ax.legend(fontsize=16, framealpha=0.9, edgecolor='black', loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved figure: {save_path}")


def plot_umap_comparison(embeddings_dict, labels, save_dir, category_names):
    """Plot a 2x2 comparison of UMAP embeddings across four modalities."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('white')

    modalities = ['fp', 'seq', 'graph', 'fused']
    titles = ['Fingerprint', 'Sequence', 'Graph', 'Fused Features']
    colors = ["#F1CF81", "#8DB4E3", "#F1918F"]

    for idx, (modality, title) in enumerate(zip(modalities, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        ax.set_facecolor('white')

        embeddings = embeddings_dict[modality]

        for i, (cat_name, color) in enumerate(zip(category_names, colors)):
            mask = labels == i
            if mask.sum() > 0:
                ax.scatter(
                    embeddings[mask, 0],
                    embeddings[mask, 1],
                    c=color,
                    label=cat_name,
                    alpha=0.6,
                    s=25,
                )

        ax.set_xlabel('UMAP 1', fontsize=16, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.tick_params(labelsize=10, width=1.5, length=5)
        ax.legend(fontsize=16, framealpha=0.9, edgecolor='black', loc='best')

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'umap_comparison_2x2.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved comparison figure: {save_path}")


def compute_umap_embeddings(features_dict, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Compute UMAP embeddings for all modalities.

    Args:
        features_dict: Feature dictionary.
        n_neighbors: Number of UMAP neighbors.
        min_dist: Minimum UMAP distance.
        metric: Distance metric.

    Returns:
        Dictionary of UMAP embeddings.
    """
    embeddings_dict = {}

    for modality, features in features_dict.items():
        print(f"\nComputing UMAP embedding for {modality.upper()}...")
        print(f"  Feature shape: {features.shape}")

        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            n_components=2,
            verbose=False,
        )

        embeddings = reducer.fit_transform(features)
        embeddings_dict[modality] = embeddings

        print(f"  UMAP embedding completed: {embeddings.shape}")

    return embeddings_dict


def analyze_cluster_quality(embeddings_dict, labels, save_dir):
    """Analyze clustering quality and save the results."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    results = []

    for modality, embeddings in embeddings_dict.items():
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        calinski = calinski_harabasz_score(embeddings, labels)

        results.append({
            'Modality': modality.upper(),
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Calinski-Harabasz Index': calinski,
        })

        print(f"\n{modality.upper()} cluster quality:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {calinski:.4f} (higher is better)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, 'cluster_quality_metrics.csv'), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    metrics = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
    colors_bar = ["#96D6D1", "#64C5DB", "#7BA8E0", "#EE817F"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = results_df[metric].values
        bars = ax.bar(results_df['Modality'], values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=16, fontweight='bold')
        ax.set_xlabel('Modality', fontsize=16, fontweight='bold')
        ax.tick_params(labelsize=10)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
            )

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'cluster_quality_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved cluster-quality comparison figure: {save_path}")

    return results_df


def main(model_path, test_file, output_dir, batch_size=128):
    """
    Run UMAP visualization analysis.

    Args:
        model_path: Path to the trained model.
        test_file: Test data file.
        output_dir: Output directory.
        batch_size: Batch size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 80}")
    print("UMAP visualization analysis - multimodal feature comparison")
    print(f"{'=' * 80}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Data: {test_file}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading test data...")
    test_set = MolDataset(test_file)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test set size: {len(test_set)}")

    print(f"\n{'=' * 80}")
    print("Step 1: Extract multimodal features")
    print(f"{'=' * 80}")
    extractor = FeatureExtractor(model_path, device)
    features_dict, labels_reg, labels_cls = extractor.extract_features(test_loader)

    print("\nExtracted feature shapes:")
    for modality, features in features_dict.items():
        print(f"  {modality.upper()}: {features.shape}")

    print(f"\n{'=' * 80}")
    print("Step 2: Categorize permeability")
    print(f"{'=' * 80}")
    permeability_categories, category_names = categorize_permeability(labels_reg, method='quantile')

    print(f"Permeability range: [{labels_reg.min():.3f}, {labels_reg.max():.3f}]")
    print("Category distribution:")
    for i, name in enumerate(category_names):
        count = (permeability_categories == i).sum()
        percentage = count / len(permeability_categories) * 100
        print(f"  {name}: {count} ({percentage:.1f}%)")

    print(f"\n{'=' * 80}")
    print("Step 3: Compute UMAP embeddings")
    print(f"{'=' * 80}")
    embeddings_dict = compute_umap_embeddings(
        features_dict,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
    )

    print(f"\n{'=' * 80}")
    print("Step 4: Plot UMAP visualizations")
    print(f"{'=' * 80}")

    modality_titles = {
        'fp': 'UMAP: Fingerprint Features',
        'seq': 'UMAP: Sequence Features',
        'graph': 'UMAP: Graph Features',
        'fused': 'UMAP: Fused Features',
    }

    for modality, title in modality_titles.items():
        save_path = os.path.join(output_dir, f'umap_{modality}.png')
        plot_umap_single(embeddings_dict[modality], permeability_categories, title, save_path, category_names)

    plot_umap_comparison(embeddings_dict, permeability_categories, output_dir, category_names)

    print(f"\n{'=' * 80}")
    print("Step 5: Analyze clustering quality")
    print(f"{'=' * 80}")
    cluster_results = analyze_cluster_quality(embeddings_dict, permeability_categories, output_dir)

    print(f"\n{'=' * 80}")
    print("Step 6: Save result data")
    print(f"{'=' * 80}")

    for modality, embeddings in embeddings_dict.items():
        df = pd.DataFrame({
            'UMAP1': embeddings[:, 0],
            'UMAP2': embeddings[:, 1],
            'Permeability_Value': labels_reg,
            'Permeability_Category': permeability_categories,
            'Category_Name': [category_names[i] for i in permeability_categories],
            'Classification_Label': labels_cls,
        })
        save_path = os.path.join(output_dir, f'umap_coordinates_{modality}.csv')
        df.to_csv(save_path, index=False)
        print(f"Saved {modality.upper()} UMAP coordinates: {save_path}")

    print(f"\n{'=' * 80}")
    print("UMAP visualization analysis completed.")
    print(f"{'=' * 80}")
    print("\nGenerated files:")
    print("  1. umap_fp.png - fingerprint feature UMAP")
    print("  2. umap_seq.png - sequence feature UMAP")
    print("  3. umap_graph.png - graph feature UMAP")
    print("  4. umap_fused.png - fused feature UMAP")
    print("  5. umap_comparison_2x2.png - 2x2 comparison plot")
    print("  6. cluster_quality_metrics.csv - cluster-quality metrics")
    print("  7. cluster_quality_comparison.png - cluster-quality comparison plot")
    print("  8. umap_coordinates_*.csv - UMAP coordinate data")
    print(f"\nAll results saved in: {output_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)

    model_path = os.path.join(base_dir, "new_result", "01weight_best_result", "fold_5", "best_model.pth")

    test_file = os.path.join(base_dir, "all_data_split2", "folds", "fold_5_test.csv")

    output_dir = os.path.join(base_dir, "umap_visualization_results")

    if not os.path.exists(model_path):
        print(f"Error: model file not found: {model_path}")
        exit(1)

    if not os.path.exists(test_file):
        print(f"Error: test file not found: {test_file}")
        exit(1)

    main(model_path, test_file, output_dir, batch_size=128)
