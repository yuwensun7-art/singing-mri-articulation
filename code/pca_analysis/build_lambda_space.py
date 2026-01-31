"""
build_lambda_space.py

Construct a low-dimensional articulatory space (λ-space) using PCA
on frame-level tongue kinematic data.

This script reproduces Experiment 2.
"""

import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = "../data/frame_level/"
OUTPUT_DIR = "../data/derived/"

vowels = ["a", "e", "i", "o", "u"]

dfs = []

for v in vowels:
    path = os.path.join(DATA_DIR, f"{v}-F1-F2.xlsx")
    df = pd.read_excel(path)
    df["vowel"] = v
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

X = data[
    [
        "tongue_tip_rel_x",
        "tongue_tip_rel_y",
        "tongue_dorsum_rel_x",
        "tongue_dorsum_rel_y",
        "tongue_root_rel_x",
        "tongue_root_rel_y",
    ]
].values

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
lambda_coords = pca.fit_transform(X_scaled)

data["lambda1"] = lambda_coords[:, 0]
data["lambda2"] = lambda_coords[:, 1]

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "Experiment2_lambda_F1F2_frame_level.xlsx")
data.to_excel(out_path, index=False)
