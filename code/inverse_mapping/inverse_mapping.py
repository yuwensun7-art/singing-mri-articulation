"""
inverse_mapping.py

Estimate articulatory λ-coordinates from observed acoustic targets
using constrained optimization (Experiment 3).
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

df_mri = pd.read_excel(
    "../data/derived/Experiment2_lambda_F1F2_frame_level.xlsx"
)

lambda_cloud = df_mri[["lambda1", "lambda2"]].dropna()

bounds = [
    (lambda_cloud["lambda1"].min(), lambda_cloud["lambda1"].max()),
    (lambda_cloud["lambda2"].min(), lambda_cloud["lambda2"].max()),
]

df_six = pd.read_excel("../data/acoustic/six-male.xlsx")

def F1_hat(lam):
    return 561.23 - 32.52 * lam[0] + 28.65 * lam[1]

def F2_hat(lam):
    return 1084.86 + 90.92 * lam[0] - 165.44 * lam[1]

def objective(lam, F1_obs, F2_obs, alpha=1e-4):
    return (
        (F1_hat(lam) - F1_obs) ** 2
        + (F2_hat(lam) - F2_obs) ** 2
        + alpha * np.sum(lam ** 2)
    )

results = []

for _, r in df_six.iterrows():
    best = None
    best_val = np.inf

    for _ in range(20):
        init = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        )

        res = minimize(
            objective,
            init,
            args=(r["F1"], r["F2"]),
            bounds=bounds,
            method="L-BFGS-B"
        )

        if res.fun < best_val:
            best_val = res.fun
            best = res.x

    results.append({
        "singer": r["singer"],
        "vowel": r["vowel"],
        "lambda1_hat": best[0],
        "lambda2_hat": best[1],
    })

pd.DataFrame(results).to_excel(
    "../data/derived/Experiment3_inverse_lambda_results.xlsx",
    index=False
)
