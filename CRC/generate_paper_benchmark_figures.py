"""
=============================================================================
Chemical Reservoir Computing: Paper-Style Benchmark Figures
=============================================================================
Reproduces the exact visual representation from Faulon et al. (Fig. 6):
  - 8 Benchmark Tasks:
      3 Linear:   AND, OR, LINEAR
      5 Nonlinear: XOR, CIRCLE, SINE, CONCENTRIC, DOTS
  - Each task saved as a SEPARATE individual figure (no subplots):
      * Upper panel: 2D scatter plot (x1 vs x2) with shaded ground-truth
        decision region, dashed quadrant axes, and red/blue classified points.
      * Lower panel: Acc comparison bar chart with error bars across:
          - SVM (light gray)
          - MLP (medium gray)
          - XGB (dark gray)
          - Conventional RC (Fixed-prior, light green)
          - Physical RC (Chemical Reservoir, dark green)
  - Plus a separate Summary Boxplot figure (matching panel b).
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
DATA_DIR_S4_R1 = "CRC/results/stage04/stage4_aligned_features.csv"
DATA_DIR_S4_R2 = "CRC/results/stage04/stage4_run2_aligned_features.csv"
DATA_DIR_S6 = "CRC/results/stage06/stage6_aligned_features.csv"

OUT_DIR = "CRC/results/logic_suite/paper_style_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Visual styling
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9
})

# Colors matching the reference screenshot
COLOR_SVM = "#e0e0e0"       # light gray
COLOR_MLP = "#b3b3b3"       # medium gray
COLOR_XGB = "#595959"       # dark gray
COLOR_CONV_RC = "#c7e9c0"   # light green
COLOR_PHYS_RC = "#238b45"   # dark green

COLOR_CLASS_0 = "#2b5c8f"   # deep blue for Class 0
COLOR_CLASS_1 = "#b22222"   # deep red for Class 1
COLOR_SHADE = "#eedede"     # light reddish-gray fill for decision region
COLOR_EDGE = "#d9534f"      # region boundary edge

# -----------------------------------------------------------------------------
# 1. Conventional ESN Reservoir Implementation
# -----------------------------------------------------------------------------
class EchoStateNetwork:
    """Conventional RC (Fixed-prior) recurrent reservoir."""
    def __init__(self, n_res=50, spectral_radius=0.95, leak_rate=0.4, seed=42):
        self.n_res = n_res
        self.leak_rate = leak_rate
        rng = np.random.RandomState(seed)
        # Random input weights
        self.Win = rng.uniform(-0.5, 0.5, (n_res, 2))
        # Random recurrent reservoir matrix
        W = rng.normal(0, 1, (n_res, n_res))
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius)
        self.readout = RidgeClassifier(alpha=1.0)

    def _transform(self, X_seq):
        states = []
        s = np.zeros(self.n_res)
        for u in X_seq:
            s = (1.0 - self.leak_rate) * s + self.leak_rate * np.tanh(np.dot(self.Win, u) + np.dot(self.W, s))
            states.append(s.copy())
        return np.array(states)

    def fit(self, X, y):
        H = self._transform(X)
        self.readout.fit(H, y)
        return self

    def predict(self, X):
        H = self._transform(X)
        return self.readout.predict(H)


# -----------------------------------------------------------------------------
# 2. Ingest Physical Chemical Data & Generate Benchmark Coordinates
# -----------------------------------------------------------------------------
def prepare_data():
    df4_1 = pd.read_csv(DATA_DIR_S4_R1)
    df4_2 = pd.read_csv(DATA_DIR_S4_R2)
    df6 = pd.read_csv(DATA_DIR_S6)

    # Active steps (discard 30 washout steps)
    act4_1 = (df4_1["step"] >= 30).values
    act4_2 = (df4_2["step"] >= 30).values
    act6 = (df6["step"] >= 30).values

    # Physical features: 6 taps of r_555_480 + time_drift
    taps = [10, 20, 30, 40, 50, 60]
    fcols = [f"r_555_480_tap{t}" for t in taps] + ["time_drift"]

    # Combine Stage 4 & Stage 6 active data
    df_all = pd.concat([df4_1[act4_1], df4_2[act4_2], df6[act6]], ignore_index=True)

    # Normalize doses to 2D coordinates [-1, 1]
    # Input x1 = u_vol(t-1), x2 = u_vol(t)
    u_vol = df_all["u_vol"].values
    max_abs = max(abs(u_vol.min()), abs(u_vol.max()), 0.1)

    # Synthesize dense grid of 2D coordinates for visual scatter (matching paper N=400)
    rng = np.random.RandomState(42)
    N_pts = 400
    x1_synth = rng.uniform(-1.0, 1.0, N_pts)
    x2_synth = rng.uniform(-1.0, 1.0, N_pts)
    X_2d = np.column_stack([x1_synth, x2_synth])

    # Physical reservoir state representations:
    X_phys = df_all[fcols].values
    # Also extract corresponding (x1, x2) from active dataset
    x1_phys = np.roll(u_vol / max_abs, 1)
    x2_phys = u_vol / max_abs
    X_phys_2d = np.column_stack([x1_phys, x2_phys])

    return X_2d, X_phys_2d, X_phys


# -----------------------------------------------------------------------------
# 3. Ground-Truth Functions for the 8 Benchmark Tasks
# -----------------------------------------------------------------------------
def get_task_ground_truth(X, task_name):
    x1 = X[:, 0]
    x2 = X[:, 1]
    if task_name == "AND":
        # Upper-right quadrant
        y = ((x1 > 0.0) & (x2 > 0.0)).astype(int)
    elif task_name == "OR":
        # All quadrants except bottom-left
        y = ((x1 > 0.0) | (x2 > 0.0)).astype(int)
    elif task_name == "LINEAR":
        # Diagonal half-space
        y = (x2 > -x1).astype(int)
    elif task_name == "XOR":
        # Alternating quadrants (checkerboard)
        y = (((x1 > 0.0) & (x2 <= 0.0)) | ((x1 <= 0.0) & (x2 > 0.0))).astype(int)
    elif task_name == "CIRCLE":
        # Central circular disc
        y = (x1**2 + x2**2 <= 0.55**2).astype(int)
    elif task_name == "SINE":
        # Sine wave boundary
        y = (x2 > 0.45 * np.sin(3.14 * x1) - 0.1).astype(int)
    elif task_name == "CONCENTRIC":
        # Annulus / concentric ring
        r2 = x1**2 + x2**2
        y = ((r2 >= 0.22**2) & (r2 <= 0.72**2)).astype(int)
    elif task_name == "DOTS":
        # Two separated circular blobs
        d1 = (x1 - 0.45)**2 + (x2 - 0.45)**2 <= 0.32**2
        d2 = (x1 + 0.45)**2 + (x2 + 0.45)**2 <= 0.32**2
        y = (d1 | d2).astype(int)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    return y


def get_grid_mask(task_name, resolution=200):
    """Dense 2D grid mask for plotting the shaded decision region."""
    xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, resolution), np.linspace(-1.0, 1.0, resolution))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    z = get_task_ground_truth(grid_pts, task_name)
    return xx, yy, z.reshape(xx.shape)


# -----------------------------------------------------------------------------
# 4. Model Training & Cross-Validation Engine
# -----------------------------------------------------------------------------
def evaluate_models_on_task(task_name, X_2d, X_phys_2d, X_phys):
    # Targets on 2D synthetic points
    y_synth = get_task_ground_truth(X_2d, task_name)
    # Targets on physical dataset
    y_phys = get_task_ground_truth(X_phys_2d, task_name)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. SVM (RBF kernel)
    svm_accs = []
    for tr, te in kf.split(X_2d):
        m = SVC(kernel="rbf", C=1.0)
        m.fit(X_2d[tr], y_synth[tr])
        svm_accs.append(m.score(X_2d[te], y_synth[te]))

    # 2. MLP
    mlp_accs = []
    for tr, te in kf.split(X_2d):
        m = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=42)
        m.fit(X_2d[tr], y_synth[tr])
        mlp_accs.append(m.score(X_2d[te], y_synth[te]))

    # 3. XGB (Gradient Boosting)
    xgb_accs = []
    for tr, te in kf.split(X_2d):
        m = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        m.fit(X_2d[tr], y_synth[tr])
        xgb_accs.append(m.score(X_2d[te], y_synth[te]))

    # 4. Conventional RC (Fixed-prior ESN)
    conv_accs = []
    for tr, te in kf.split(X_2d):
        m = EchoStateNetwork(n_res=50, spectral_radius=0.95, seed=42)
        m.fit(X_2d[tr], y_synth[tr])
        conv_accs.append(m.readout.score(m._transform(X_2d[te]), y_synth[te]))

    # 5. Physical Chemical RC (Readout on physical optical states)
    phys_accs = []
    scaler = StandardScaler()
    X_phys_s = scaler.fit_transform(X_phys)
    for tr, te in kf.split(X_phys_s):
        m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=500)
        m.fit(X_phys_s[tr], y_phys[tr])
        phys_accs.append(m.score(X_phys_s[te], y_phys[te]))

    scores = {
        "SVM": (float(np.mean(svm_accs)), float(np.std(svm_accs) / np.sqrt(5))),
        "MLP": (float(np.mean(mlp_accs)), float(np.std(mlp_accs) / np.sqrt(5))),
        "XGB": (float(np.mean(xgb_accs)), float(np.std(xgb_accs) / np.sqrt(5))),
        "Conventional RC": (float(np.mean(conv_accs)), float(np.std(conv_accs) / np.sqrt(5))),
        "Physical RC": (float(np.mean(phys_accs)), float(np.std(phys_accs) / np.sqrt(5))),
    }

    raw_scores = {
        "SVM": svm_accs,
        "MLP": mlp_accs,
        "XGB": xgb_accs,
        "Conventional RC": conv_accs,
        "Physical RC": phys_accs
    }

    return scores, raw_scores, y_synth


# -----------------------------------------------------------------------------
# 5. Render Individual Standalone Figures (No Subplots)
# -----------------------------------------------------------------------------
def render_individual_task_figure(task_name, X_2d, y_synth, scores, fpath):
    """
    Renders a single standalone figure for task_name matching the paper's exact layout:
      - Upper box: 2D scatter with shaded region
      - Lower box: Acc comparison bar chart with error bars
    """
    fig = plt.figure(figsize=(4.2, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.95], hspace=0.32)

    # -------------------------------------------------------------------------
    # Upper Box: 2D Scatter Plot with Shaded Region
    # -------------------------------------------------------------------------
    ax_top = fig.add_subplot(gs[0])

    # 1. Shaded ground-truth region
    xx, yy, z_mask = get_grid_mask(task_name)
    ax_top.contourf(xx, yy, z_mask, levels=[-0.5, 0.5, 1.5], colors=["#ffffff", COLOR_SHADE], alpha=0.9)
    ax_top.contour(xx, yy, z_mask, levels=[0.5], colors=[COLOR_EDGE], linewidths=1.2, linestyles="--")

    # 2. Quadrant dashed axes
    ax_top.axhline(0.0, color="#8c8c8c", linestyle=":", linewidth=0.9)
    ax_top.axvline(0.0, color="#8c8c8c", linestyle=":", linewidth=0.9)

    # 3. Scatter points
    mask0 = (y_synth == 0)
    mask1 = (y_synth == 1)
    ax_top.scatter(X_2d[mask0, 0], X_2d[mask0, 1], c=COLOR_CLASS_0, s=12, alpha=0.85, edgecolors="none")
    ax_top.scatter(X_2d[mask1, 0], X_2d[mask1, 1], c=COLOR_CLASS_1, s=12, alpha=0.85, edgecolors="none")

    ax_top.set_xlim(-1.05, 1.05)
    ax_top.set_ylim(-1.05, 1.05)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title(task_name, fontweight="bold", fontsize=13, pad=8)
    for spine in ax_top.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.0)

    # -------------------------------------------------------------------------
    # Lower Box: Acc Comparison Bar Chart
    # -------------------------------------------------------------------------
    ax_bot = fig.add_subplot(gs[1])

    models = ["SVM", "MLP", "XGB", "Conventional RC", "Physical RC"]
    means = [scores[m][0] for m in models]
    sems = [scores[m][1] for m in models]
    colors = [COLOR_SVM, COLOR_MLP, COLOR_XGB, COLOR_CONV_RC, COLOR_PHYS_RC]

    x_pos = np.arange(len(models))
    bars = ax_bot.bar(x_pos, means, yerr=sems, capsize=3, color=colors, edgecolor="#404040", linewidth=0.8, width=0.68)

    ax_bot.set_ylabel("Acc", fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0.5, 1.02)
    ax_bot.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_bot.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax_bot.set_xticks([])
    ax_bot.grid(True, axis="y", linestyle="-", color="#f0f0f0", linewidth=0.8, zorder=0)
    for bar in bars:
        bar.set_zorder(3)

    for spine in ["top", "right"]:
        ax_bot.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_bot.spines[spine].set_color("#666666")
        ax_bot.spines[spine].set_linewidth(0.9)

    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    print(f"  Saved individual figure: {fpath}")


# -----------------------------------------------------------------------------
# 6. Render Summary Boxplot Figure (Matching Panel b)
# -----------------------------------------------------------------------------
def render_summary_boxplots(all_task_scores, fpath):
    """
    Renders the overall boxplot comparison matching panel (b) across all 8 tasks.
    """
    fig, ax = plt.subplots(figsize=(5.2, 6.0))

    models = ["SVM", "MLP", "XGB", "Conventional RC", "Physical RC"]
    model_labels = ["SVM", "MLP", "XGB", "Conventional RC\n(Fixed-prior)", "Physical RC\n(Chemical)"]
    box_colors = [COLOR_SVM, COLOR_MLP, COLOR_XGB, COLOR_CONV_RC, COLOR_PHYS_RC]

    # Gather mean accuracy across all 8 tasks for each model
    data_by_model = []
    for m in models:
        scores_m = [all_task_scores[t][m][0] for t in all_task_scores]
        data_by_model.append(scores_m)

    bp = ax.boxplot(
        data_by_model,
        positions=np.arange(len(models)),
        widths=0.62,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="x", markeredgecolor="black", markersize=6),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="#404040", linewidth=1.0),
        capprops=dict(color="#404040", linewidth=1.0),
        flierprops=dict(marker="o", markersize=4, alpha=0.5)
    )

    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_edgecolor("#404040")
        patch.set_linewidth(1.0)
        patch.set_alpha(0.85)

    # Overlay individual data points
    rng = np.random.RandomState(42)
    for i, pts in enumerate(data_by_model):
        jitter = rng.normal(0, 0.04, size=len(pts))
        ax.scatter(np.full_like(pts, i) + jitter, pts, color="#2b2b2b", s=14, alpha=0.6, zorder=4)

    ax.set_ylabel("Acc", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9.5)
    ax.grid(True, axis="y", linestyle="-", color="#f0f0f0", linewidth=0.8, zorder=0)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(1.0)

    ax.set_title("Overall Benchmark Accuracy Distribution", fontweight="bold", fontsize=12, pad=12)

    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    print(f"  Saved summary boxplot figure: {fpath}")


# -----------------------------------------------------------------------------
# 7. Render Standalone Color Legend Card
# -----------------------------------------------------------------------------
def render_legend_card(fpath):
    """Renders a standalone visual legend card matching the paper key."""
    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    models = [
        ("SVM", COLOR_SVM),
        ("MLP", COLOR_MLP),
        ("XGB", COLOR_XGB),
        ("Conventional RC (Fixed-prior)", COLOR_CONV_RC),
        ("Physical RC (Chemical)", COLOR_PHYS_RC),
    ]

    for idx, (label, col) in enumerate(models):
        y_pos = len(models) - 1 - idx
        ax.add_patch(plt.Rectangle((0.1, y_pos - 0.28), 0.25, 0.55, facecolor=col, edgecolor="#404040", lw=1.0))
        ax.text(0.42, y_pos, label, va="center", ha="left", fontsize=10.5, fontweight="bold" if "Physical" in label else "normal")

    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.8, len(models) - 0.2)
    ax.axis("off")
    ax.set_title("Model Color Codes", fontweight="bold", fontsize=12, pad=8)

    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    print(f"  Saved legend card: {fpath}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("GENERATING FAULON ET AL. PAPER-STYLE BENCHMARK FIGURES")
    print("=" * 70)

    X_2d, X_phys_2d, X_phys = prepare_data()

    tasks = [
        # 3 Linear
        "AND",
        "OR",
        "LINEAR",
        # 5 Nonlinear
        "XOR",
        "CIRCLE",
        "SINE",
        "CONCENTRIC",
        "DOTS"
    ]

    all_scores = {}
    csv_rows = []

    for task in tasks:
        print(f"\nProcessing Task: {task} ...")
        scores, raw_scores, y_synth = evaluate_models_on_task(task, X_2d, X_phys_2d, X_phys)
        all_scores[task] = scores

        # Save individual standalone figure
        fpath = os.path.join(OUT_DIR, f"fig_benchmark_{task}.png")
        render_individual_task_figure(task, X_2d, y_synth, scores, fpath)

        for m, (mean_acc, sem_acc) in scores.items():
            csv_rows.append({
                "task": task,
                "model": m,
                "mean_acc": mean_acc,
                "sem_acc": sem_acc
            })
            print(f"   {m:<16s}: Acc = {mean_acc*100:5.1f}% ± {sem_acc*100:4.1f}%")

    # Save summary boxplot figure (matching panel b)
    box_fpath = os.path.join(OUT_DIR, "fig_benchmark_summary_boxplots.png")
    render_summary_boxplots(all_scores, box_fpath)

    # Save color legend card
    legend_fpath = os.path.join(OUT_DIR, "fig_benchmark_color_legend.png")
    render_legend_card(legend_fpath)

    # Save CSV table
    df_out = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUT_DIR, "paper_style_benchmark_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics CSV: {csv_path}")

    print("\nALL PAPER-STYLE INDIVIDUAL FIGURES GENERATED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
