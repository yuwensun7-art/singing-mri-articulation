"""
compute_vowel_means.py

Compute vowel-level mean articulatory positions from frame-level
tracker outputs. All coordinates are normalized relative to a
head reference point.

This script generates summary statistics reported in Table 1.
"""

import pandas as pd
import os

DATA_DIR = "../data/articulatory_tables/"
OUTPUT_DIR = "../data/derived/"

vowel_files = {
    "a": "a-vowel.xlsx",
    "e": "e-vowel.xlsx",
    "i": "i-vowel.xlsx",
    "o": "o-vowel.xlsx",
    "u": "u-vowel.xlsx",
}

results = []

for vowel, filename in vowel_files.items():
    path = os.path.join(DATA_DIR, filename)

    # Tracker output with two-row headers
    df = pd.read_excel(path, header=[0, 1])

    head_x = df[("head-ref", "x")]
    head_y = df[("head-ref", "y")]

    def rel(col):
        return df[col] - head_x if col[1] == "x" else df[col] - head_y

    results.append({
        "vowel": vowel,
        "n_frames": len(df),

        "tongue_root_rel_x_mean": (df[("tongue root", "x")] - head_x).mean(),
        "tongue_root_rel_y_mean": (df[("tongue root", "y")] - head_y).mean(),

        "tongue_tip_rel_x_mean": (df[("tongue tip", "x")] - head_x).mean(),
        "tongue_tip_rel_y_mean": (df[("tongue tip", "y")] - head_y).mean(),

        "tongue_dorsum_rel_x_mean": (df[("tongue dorsum", "x")] - head_x).mean(),
        "tongue_dorsum_rel_y_mean": (df[("tongue dorsum", "y")] - head_y).mean(),

        "palate_rel_x_mean": (df[("soft palate", "x")] - head_x).mean(),
        "palate_rel_y_mean": (df[("soft palate", "y")] - head_y).mean(),
    })

table1 = pd.DataFrame(results)

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "vowel_articulatory_means.xlsx")
table1.to_excel(out_path, index=False)
