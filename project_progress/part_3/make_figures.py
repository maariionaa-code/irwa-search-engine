"""Create the report figures from the CSV files in ``results/``.

Usage from the repository root (after evaluate_methods and compare_rankings):
    python -m project_progress.part_3.make_figures
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from project_progress.part_3.compare_rankings import AGREEMENT_CSV  # noqa: E402
from project_progress.part_3.evaluate_methods import METRICS_CSV  # noqa: E402
from project_progress.part_3.run_rankings import RESULTS_DIR  # noqa: E402


FIGURES_DIR = RESULTS_DIR.parent / "figures"
METHOD_ORDER = ["tfidf", "bm25", "our_score", "intent_score", "word2vec", "doc2vec"]
LABELS = {"tfidf": "TF-IDF", "bm25": "BM25", "our_score": "Our score", "intent_score": "Intent score",
          "word2vec": "word2vec", "doc2vec": "Doc2Vec"}


def p_at_10_by_query(metrics: pd.DataFrame):
    """Grouped bars: P@10 of each method for each query."""
    data = metrics[metrics["k"] == 10].pivot(
        index="query_id", columns="method", values="P@K")
    methods = [m for m in METHOD_ORDER if m in data.columns]
    x = np.arange(len(data.index))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, method in enumerate(methods):
        ax.bar(x + i * width - 0.4 + width / 2, data[method], width,
               label=LABELS[method])
    ax.set_xticks(x, data.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision@10")
    ax.legend(ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "p10_by_query.png", dpi=200)
    plt.close(fig)


def kendall_heatmap(agreement: pd.DataFrame):
    """Mean Kendall tau between every pair of methods."""
    methods = [m for m in METHOD_ORDER
               if m in set(agreement["method_a"]) | set(agreement["method_b"])]
    mean = agreement.groupby(["method_a", "method_b"])["kendall_tau"].mean()
    matrix = np.eye(len(methods))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if (a, b) in mean.index:
                matrix[i, j] = matrix[j, i] = mean[(a, b)]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=30)
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(matrix[i, j]) > 0.6 else "black")
    fig.colorbar(image, ax=ax, label="Mean Kendall tau")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kendall_tau_heatmap.png", dpi=200)
    plt.close(fig)


def map_by_k(metrics: pd.DataFrame):
    """MAP@K (pool) of every method for K = 5, 10, 15, 20, all queries."""
    data = metrics.groupby(["method", "k"])["AP@K_pool"].mean().unstack("method")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in [m for m in METHOD_ORDER if m in data.columns]:
        ax.plot(data.index, data[method], marker="o", label=LABELS[method])
    ax.set_xticks(list(data.index))
    ax.set_xlabel("K (cut-off)")
    ax.set_ylabel("MAP@K (pool)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "map_by_k.png", dpi=200)
    plt.close(fig)


def main():
    """Write all figures to ``figures/``."""
    FIGURES_DIR.mkdir(exist_ok=True)
    metrics = pd.read_csv(METRICS_CSV)
    p_at_10_by_query(metrics)
    map_by_k(metrics)
    kendall_heatmap(pd.read_csv(AGREEMENT_CSV))
    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()