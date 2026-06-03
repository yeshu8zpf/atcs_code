import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
utilities = ["IFD", "NLL", "UFS"]
spearman = [0.413, 0.278, 0.373]
pairwise_agreement = [0.643, 0.589, 0.638]

# =========================
# Plot settings
# =========================
x = np.arange(len(utilities))
width = 0.34

fig, ax = plt.subplots(figsize=(6.2, 4.2))

bars1 = ax.bar(x - width / 2, spearman, width, label="Spearman")
bars2 = ax.bar(x + width / 2, pairwise_agreement, width, label="Pairwise agreement")

# Axis labels and title
ax.set_xlabel("Utility", fontsize=12)
ax.set_ylabel("Ranking consistency", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(utilities, fontsize=11)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=10, frameon=False)

# Add value labels
def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

add_labels(bars1)
add_labels(bars2)

# # Optional annotation to highlight NLL
# nll_idx = utilities.index("NLL")
# ax.annotate(
#     "Lowest consistency",
#     xy=(x[nll_idx], max(spearman[nll_idx], pairwise_agreement[nll_idx])),
#     xytext=(x[nll_idx] + 0.15, 0.71),
#     arrowprops=dict(arrowstyle="->", lw=1.0),
#     fontsize=10
# )

# Make layout cleaner
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

# =========================
# Save
# =========================

pdf_path = "analyze/results/ranking_consistency_bar.pdf"

plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print(f"Saved to: {pdf_path}")