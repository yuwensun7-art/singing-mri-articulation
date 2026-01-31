"""
Plot feasibility of inverse articulatory solutions
relative to MRI-derived lambda space.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pathlib import Path

# ===============================
# Paths (relative to repo root)
# ===============================
DATA_DIR = Path("data/derived")

mri_path = DATA_DIR / "Experiment2_lambda_F1F2_frame_level.xlsx"
inv_path = DATA_DIR / "Experiment3_inverse_lambda_results.xlsx"

# ===============================
# Load data
# ===============================
df_mri = pd.read_excel(mri_path)
df_inv = pd.read_excel(inv_path)

lambda_mri = df_mri[["lambda1", "lambda2"]].dropna()
lambda_hat = df_inv[["lambda1_hat", "lambda2_hat", "vowel"]]

# ===============================
# Plot
# ===============================
plt.figure(figsize=(7, 7))

# MRI lambda cloud
plt.scatter(
    lambda_mri["lambda1"],
    lambda_mri["lambda2"],
    s=8,
    c="lightgray",
    alpha=0.5,
    label="MRI-derived λ-cloud",
)

# Convex hull
if len(lambda_mri) >= 3:
    hull = ConvexHull(lambda_mri.values)
    for simplex in hull.simplices:
        plt.plot(
            lambda_mri.values[simplex, 0],
            lambda_mri.values[simplex, 1],
            color="gray",
            linewidth=1,
        )

# Inverse solutions
vowel_colors = {
    "a": "red",
    "e": "orange",
    "i": "green",
    "o": "blue",
    "u": "purple",
}

for vowel, group in lambda_hat.groupby("vowel"):
    plt.scatter(
        group["lambda1_hat"],
        group["lambda2_hat"],
        s=80,
        c=vowel_colors.get(vowel, "black"),
        edgecolors="k",
        label=f"/{vowel}/",
    )

# Cosmetics
plt.xlabel("λ₁")
plt.ylabel("λ₂")
plt.title("Feasibility of inverse solutions in MRI-derived λ-space")
plt.axis("equal")

# 去重 legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(dict(zip(labels, handles)).values(),
           dict(zip(labels, handles)).keys(),
           frameon=False)

plt.tight_layout()
plt.show()
