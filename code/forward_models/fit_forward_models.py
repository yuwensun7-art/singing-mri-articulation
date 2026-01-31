"""
Fit vowel-conditioned linear forward models:
lambda space -> F1 / F2
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ===============================
# Load data
# ===============================
DATA_DIR = Path("data/derived")
data_path = DATA_DIR / "Experiment2_lambda_F1F2_frame_level.xlsx"

df = pd.read_excel(data_path)

# Clean F1 / F2
for col in ["F1", "F2"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("Hz", "", regex=False)
        .str.strip()
        .astype(float)
    )

X_cols = ["lambda1", "lambda2"]

print("=== Vowel-conditioned articulatory → acoustic mappings ===\n")

models = {}

for vowel in sorted(df["vowel"].unique()):
    df_v = df[df["vowel"] == vowel]

    X = df_v[X_cols]
    Y_F1 = df_v["F1"]
    Y_F2 = df_v["F2"]

    model_F1 = LinearRegression().fit(X, Y_F1)
    model_F2 = LinearRegression().fit(X, Y_F2)

    models[vowel] = {"F1": model_F1, "F2": model_F2}

    print(f"--- Vowel: /{vowel}/ ---")
    print(
        f"F1 = {model_F1.intercept_:.3f} "
        f"+ {model_F1.coef_[0]:.3f}·λ1 "
        f"+ {model_F1.coef_[1]:.3f}·λ2"
    )
    print("R²(F1):", round(model_F1.score(X, Y_F1), 3))

    print(
        f"F2 = {model_F2.intercept_:.3f} "
        f"+ {model_F2.coef_[0]:.3f}·λ1 "
        f"+ {model_F2.coef_[1]:.3f}·λ2"
    )
    print("R²(F2):", round(model_F2.score(X, Y_F2), 3))
    print()
