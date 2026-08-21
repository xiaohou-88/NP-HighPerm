from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "umap_visualization_results"
MODALITIES = {
    "fp": "Fingerprint",
    "seq": "Sequence",
    "graph": "Graph",
    "fused": "Fused Features",
}
CATEGORY_COLORS = {
    "Low": "#F1CF81",
    "Mid": "#8DB4E3",
    "High": "#F1918F",
}
DEFAULT_CATEGORY_ORDER = ["Low", "Mid", "High"]


def configure_plot_style():
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


def load_umap_coordinates(input_dir):
    tables = {}
    for modality in MODALITIES:
        csv_path = input_dir / f"umap_coordinates_{modality}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        missing_columns = {"UMAP1", "UMAP2"} - set(df.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path.name} is missing required columns: {missing_text}")

        tables[modality] = df

    if not tables:
        expected = ", ".join(f"umap_coordinates_{name}.csv" for name in MODALITIES)
        raise FileNotFoundError(f"No UMAP coordinate CSV files found in {input_dir}. Expected files: {expected}")

    return tables


def get_category_labels(df):
    if "Category_Name" in df.columns:
        return df["Category_Name"].astype(str)

    if "Permeability_Category" in df.columns:
        mapping = {0: "Low", 1: "Mid", 2: "High"}
        return df["Permeability_Category"].map(mapping).fillna(df["Permeability_Category"].astype(str))

    return pd.Series(["All"] * len(df), index=df.index)


def iter_categories(labels):
    present = set(labels.unique())
    ordered = [name for name in DEFAULT_CATEGORY_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_umap_single(df, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = get_category_labels(df)
    for category in iter_categories(labels):
        mask = labels == category
        color = CATEGORY_COLORS.get(category, "#777777")
        ax.scatter(
            df.loc[mask, "UMAP1"],
            df.loc[mask, "UMAP2"],
            c=color,
            label=category,
            alpha=0.6,
            s=30,
        )

    ax.set_xlabel("UMAP 1", fontsize=16, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.tick_params(labelsize=11, width=1.5, length=5)
    ax.legend(fontsize=16, framealpha=0.9, edgecolor="black", loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_umap_comparison(tables, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax, (modality, title) in zip(axes, MODALITIES.items()):
        ax.set_facecolor("white")
        df = tables.get(modality)

        if df is None:
            ax.axis("off")
            continue

        labels = get_category_labels(df)
        for category in iter_categories(labels):
            mask = labels == category
            color = CATEGORY_COLORS.get(category, "#777777")
            ax.scatter(
                df.loc[mask, "UMAP1"],
                df.loc[mask, "UMAP2"],
                c=color,
                label=category,
                alpha=0.6,
                s=25,
            )

        ax.set_xlabel("UMAP 1", fontsize=16, fontweight="bold")
        ax.set_ylabel("UMAP 2", fontsize=16, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.tick_params(labelsize=10, width=1.5, length=5)
        ax.legend(fontsize=16, framealpha=0.9, edgecolor="black", loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def main(input_dir=DEFAULT_INPUT_DIR, output_dir=None):
    configure_plot_style()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_umap_coordinates(input_dir)

    for modality, title in MODALITIES.items():
        if modality not in tables:
            continue
        save_path = output_dir / f"umap_{modality}.png"
        plot_umap_single(tables[modality], f"UMAP: {title}", save_path)
        print(f"Saved figure: {save_path.name}")

    comparison_path = output_dir / "umap_comparison_2x2.png"
    plot_umap_comparison(tables, comparison_path)
    print(f"Saved comparison figure: {comparison_path.name}")


if __name__ == "__main__":
    main()
