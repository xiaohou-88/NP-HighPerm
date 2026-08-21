import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr


# Load prediction data.
df = pd.read_csv('fold_5_detailed_predictions.csv')

# Configure plotting defaults.
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13

# Create the main figure and subplot layout.
fig = plt.figure(figsize=(12, 12))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(
    3,
    3,
    width_ratios=[1, 4, 1],
    height_ratios=[1, 4, 1],
    hspace=0.05,
    wspace=0.05,
)

ax_main = fig.add_subplot(gs[1, 1])
ax_top = fig.add_subplot(gs[0, 1])
ax_right = fig.add_subplot(gs[1, 2])

ax_main.set_facecolor('white')
ax_top.set_facecolor('white')
ax_right.set_facecolor('white')

# Main scatter plot.
distances = np.abs(df['true_reg'] - df['pred_reg'])
max_distance = distances.max()
normalized_distances = 1 - (distances / max_distance)

scatter = ax_main.scatter(
    df['true_reg'],
    df['pred_reg'],
    alpha=0.6,
    s=100,
    c=normalized_distances,
    cmap='Blues',
    linewidth=1.0,
    vmin=0,
    vmax=1.1,
)

min_val = min(df['true_reg'].min(), df['pred_reg'].min())
max_val = max(df['true_reg'].max(), df['pred_reg'].max())
ax_main.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='#E74C3C',
    linestyle='--',
    linewidth=3.5,
    alpha=0.9,
    label='y = x',
    zorder=5,
)

ax_main.set_xlabel('True Value', fontsize=18, fontweight='bold', labelpad=12)
ax_main.set_ylabel('Predicted Value', fontsize=18, fontweight='bold', labelpad=12)

margin = 0.3
x_range = (min_val - margin, max_val + margin)
y_range = (min_val - margin, max_val + margin)
ax_main.set_xlim(x_range)
ax_main.set_ylim(y_range)

for spine in ax_main.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
ax_main.tick_params(axis='both', which='major', labelsize=14, width=2, length=7)
ax_main.set_aspect('equal', adjustable='box')

legend = ax_main.legend(
    loc='upper left',
    fontsize=15,
    framealpha=0.95,
    edgecolor='black',
    fancybox=True,
    shadow=False,
    frameon=True,
)
legend.get_frame().set_linewidth(1.5)

spearman_r, _ = spearmanr(df['true_reg'], df['pred_reg'])
metrics_text = f'$r$ = {spearman_r:.4f}'
ax_main.text(
    0.03,
    0.86,
    metrics_text,
    transform=ax_main.transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.5),
)

# Small colorbar inside the main plot.
cbar_ax = fig.add_axes([0.62, 0.3, 0.1, 0.02])
cbar = plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Prediction Accuracy', fontsize=10, fontweight='bold', labelpad=8)
cbar.ax.tick_params(labelsize=9, width=1, length=9)
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'])

# Marginal distribution for true values.
bins = 30
ax_top.hist(
    df['true_reg'],
    bins=bins,
    color='#8DADD7',
    alpha=0.7,
    edgecolor='white',
    linewidth=1.5,
    density=True,
)

ax_top.set_xlim(x_range)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_top.spines['left'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.set_xticks([])
ax_top.set_yticks([])
ax_top.tick_params(left=False, bottom=False, top=False, right=False)

# Marginal distribution for predicted values.
ax_right.hist(
    df['pred_reg'],
    bins=bins,
    orientation='horizontal',
    color="#C5B0E6",
    alpha=0.7,
    edgecolor='white',
    linewidth=1.5,
    density=True,
)

ax_right.set_ylim(y_range)
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.spines['bottom'].set_visible(False)
ax_right.set_xticks([])
ax_right.set_yticks([])
ax_right.tick_params(left=False, bottom=False, top=False, right=False)

plt.tight_layout()

plt.savefig(
    'true_vs_pred_scatter_with_marginals_clean.png',
    dpi=600,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none',
)

print("\nFigure saved as: true_vs_pred_scatter_with_marginals_clean.png")
print("\nDataset information:")
print(f"  Sample count: {len(df)}")
print(f"  True value range: [{df['true_reg'].min():.3f}, {df['true_reg'].max():.3f}]")
print(f"  Predicted value range: [{df['pred_reg'].min():.3f}, {df['pred_reg'].max():.3f}]")

r2 = r2_score(df['true_reg'], df['pred_reg'])
rmse = np.sqrt(mean_squared_error(df['true_reg'], df['pred_reg']))
mae = mean_absolute_error(df['true_reg'], df['pred_reg'])

print("\nEvaluation metrics:")
print(f"  R2 Score: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print("\nScatter color guide:")
print("  Dark blue = accurate prediction near the diagonal")
print("  Light blue = larger prediction error away from the diagonal")
print(f"  Maximum error: {max_distance:.4f}")

plt.show()
