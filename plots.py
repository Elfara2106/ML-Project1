"""Figures of the report, built from the results saved by experiments.py.

Usage:
    python plots.py               # figures 1 and 2, from results/*.tsv (instant)
    python plots.py --threshold   # also figure 3, which trains one model (~2 min)

Matplotlib is only used here, for visualization, as allowed by the project.
The figures are saved as PDF in figures/.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# Colors: first slots of a colorblind-validated categorical palette; every
# series also has its own marker / line style so the figures survive grayscale.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Largest eigenvalue of X^T X / N (special codes + one-hot features): GD on the
# MSE diverges for gamma > 2 / L
L_MSE = 31.7

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def read_tsv(path):
    """Read a tsv results file into a list of dicts (numbers converted to float)."""
    with open(path) as f:
        header = f.readline().strip().split("\t")
        rows = []
        for line in f:
            row = {}
            for key, value in zip(header, line.strip().split("\t")):
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def plot_ablation(output):
    """Figure 1: cross-validated F1 gained by each preprocessing step."""
    rows = read_tsv("results/ablation.tsv")
    names = [r["step"] for r in rows][::-1]
    means = np.array([r["f1_mean"] for r in rows])[::-1]
    stds = np.array([r["f1_std"] for r in rows])[::-1]

    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    y = np.arange(len(names))
    ax.barh(y, means, height=0.6, color=BLUE, xerr=stds, ecolor=MUTED, capsize=2)
    for yi, m, sd in zip(y, means, stds):
        ax.text(m + sd + 0.0015, yi, f"{m:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0.40, 0.455)
    ax.set_xticks([0.40, 0.42, 0.44])
    ax.set_xlabel("F1 score (5-fold CV)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved {output}")


def plot_validation_curves(output):
    """Figure 2: validation (and train) F1 as a function of each hyper-parameter."""
    rows = read_tsv("results/tuning.tsv")
    has_train = "f1_train_mean" in rows[0]
    panels = [
        ("ridge", r"Ridge regression: $\lambda$"),
        ("reg_log", r"Reg. logistic regression: $\lambda$"),
        ("mean_square_gd", r"GD on MSE: step size $\gamma$"),
        ("log", r"Logistic regression: step size $\gamma$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharey=True)
    for ax, (method, title) in zip(axes.ravel(), panels):
        r = [row for row in rows if row["method"] == method]
        x = np.array([row["value"] for row in r])
        mean = np.array([row["f1_mean"] for row in r])
        std = np.array([row["f1_std"] for row in r])

        ax.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.15, lw=0)
        ax.plot(x, mean, color=BLUE, lw=2, marker="o", ms=5, label="validation")
        if has_train:
            train = np.array([row["f1_train_mean"] for row in r])
            ax.plot(
                x, train, color=ORANGE, lw=2, ls="--", marker="s", ms=5, label="train"
            )
        best = int(np.argmax(mean))
        ax.plot(x[best], mean[best], marker="o", ms=9, mfc="none", mec=INK, mew=1.5)
        if method == "mean_square_gd":
            ax.axvline(2 / L_MSE, color=MUTED, ls=":", lw=1.2)
            ax.text(
                2 / L_MSE,
                0.395,
                r"$2/L$: divergence ",
                color=MUTED,
                fontsize=8,
                ha="right",
            )
        ax.set_xscale("log")
        ax.minorticks_off()
        if method == "mean_square_gd":
            ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 score")
    axes[0, 0].legend(loc="lower left")
    axes[0, 0].set_ylim(0.39, 0.45)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved {output}" + ("" if has_train else " (no train curve in the results)"))


def plot_threshold(output, data_path="./data"):
    """Figure 3: F1, precision and recall as a function of the decision threshold.

    Trains the final model (reg. logistic regression, special codes + one-hot)
    on 80% of the training data and evaluates on the other 20%.
    """
    # imported here so that figures 1 and 2 do not need the data
    import implementations
    import preprocessing
    from experiments import load_data

    x, y = load_data(data_path)
    tr, va = preprocessing.train_val_split(len(y))
    tx_tr, tx_va = preprocessing.build_features(
        x[tr], x[va], fix_codes=True, one_hot=True
    )
    w, _ = implementations.reg_logistic_regression(
        preprocessing.to_01(y[tr]), tx_tr, 1e-4, np.zeros(tx_tr.shape[1]), 1000, 0.5
    )
    threshold, _ = preprocessing.best_threshold(tx_tr, w, y[tr])

    proba = implementations.sigmoid(tx_va.dot(w))
    cuts = np.linspace(0.02, 0.8, 157)
    precision, recall, f1 = [], [], []
    for c in cuts:
        pred = proba > c
        tp = np.sum(pred & (y[va] == 1))
        precision.append(tp / max(pred.sum(), 1))
        recall.append(tp / np.sum(y[va] == 1))
        f1.append(preprocessing.f1_score(y[va], np.where(pred, 1, -1)))

    chosen = implementations.sigmoid(threshold)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(cuts, f1, color=BLUE, lw=2, label="F1")
    ax.plot(cuts, precision, color=ORANGE, lw=2, ls="--", label="precision")
    ax.plot(cuts, recall, color=AQUA, lw=2, ls="-.", label="recall")
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.text(0.505, 0.9, "default 0.5", color=MUTED, fontsize=8)
    ax.axvline(chosen, color=INK, lw=1.2)
    ax.text(chosen + 0.005, 0.9, f"chosen {chosen:.2f}", color=INK, fontsize=8)
    ax.set_xlabel("Threshold on the predicted probability")
    ax.set_ylabel("Score (validation)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description="Figures of the report")
    parser.add_argument(
        "--threshold",
        action="store_true",
        help="Also make figure 3 (trains one model on the data, ~2 min)",
    )
    parser.add_argument("--data_path", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    plot_ablation("figures/ablation.pdf")
    plot_validation_curves("figures/validation_curves.pdf")
    if args.threshold:
        plot_threshold("figures/threshold.pdf", args.data_path)


if __name__ == "__main__":
    main()
