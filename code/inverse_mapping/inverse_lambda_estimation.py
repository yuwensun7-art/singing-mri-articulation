"""
Inverse mapping from acoustics (F1, F2)
to articulatory lambda space using
vowel-conditioned forward models.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# ===============================
# Load MRI lambda cloud
# ===============================
DATA_DIR = Path("data/derived")

mri_path = DATA_DIR / "Experiment2_lambda_F1F2_frame_level.xlsx"
six_path = DATA_DIR / "six-male.xlsx"

df_mri = pd.read_excel(mri_path)
lambda_cloud = df_mri[["lambda1", "lambda2"]].dropna()

lam1_min, lam1_max = lambda_cloud["lambda1"].min(), lambda_cloud["lambda1"].max()
lam2_min, lam2_max = lambda_cloud["lambda2"].min(), lambda_cloud["lambda2"].max()
bounds = [(lam1_min, lam1_max), (lam2_min, lam2_max)]

df_six = pd.read_excel(six_path)

# ===============================
# Forward models (from Experiment 2)
# ===============================
forward_models = {
    "a": {
        "F1": lambda lam: 662.675 - 9.497 * lam[0] + 25.011 * lam[1],
        "F2": lambda lam: 983.492 - 5.815 * lam[0] - 11.009 * lam[1],
    },
    "e": {
        "F1": lambda lam: 659.844 + 42.046 * lam[0] + 29.673 * lam[1],
        "F2": lambda lam: 1323.773 + 26.290 * lam[0] - 25.893 * lam[1],
    },
    "i": {
        "F1": lambda lam: 568.518 - 30.050 * lam[0] + 61.004 * lam[1],
        "F2": lambda lam: 1687.315 + 32.946 * lam[0] + 13.794 * lam[1],
    },
    "o": {
        "F1": lambda lam: 358.618 + 5.678 * lam[0] - 114.434 * lam[1],
        "F2": lambda lam: 633.124 - 43.681 * lam[0] - 14.039 * lam[1],
    },
    "u": {
        "F1": lambda lam: 387.717 + 16.268 * lam[0] - 8.379 * lam[1],
        "F2": lambda lam: 822.532 + 17.710 * lam[0] - 58.213 * lam[1],
    },
}

def inverse_objective(lam, F1_obs, F2_obs, vowel, alpha=1e-4):
    model = forward_models[vowel]
    pred = np.array([model["F1"](lam), model["F2"](lam)])
    obs = np.array([F1_obs, F2_obs])
    return np.sum((pred - obs) ** 2) + alpha * np.sum(lam ** 2)

# ===============================
# Optimization
# ===============================
results = []

for _, row in df_six.iterrows():
    best_val = np.inf
    best_sol = None

    for _ in range(20):
        init = np.array([
            np.random.uniform(lam1_min, lam1_max),
            np.random.uniform(lam2_min, lam2_max),
        ])

        res = minimize(
            inverse_objective,
            init,
            args=(row["F1"], row["F2"], row["vowel"]),
            method="L-BFGS-B",
            bounds=bounds,
        )

        if res.fun < best_val:
            best_val = res.fun
            best_sol = res.x

    results.append({
        "singer": row["singer"],
        "vowel": row["vowel"],
        "F1_obs": row["F1"],
        "F2_obs": row["F2"],
        "lambda1_hat": best_sol[0],
        "lambda2_hat": best_sol[1],
        "loss": best_val,
    })

df_inv = pd.DataFrame(results)
print(df_inv)
