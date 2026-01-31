"""
articulatory_to_acoustic.py

Fit linear models mapping articulatory λ-coordinates
to acoustic formants (F1, F2).
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_excel(
    "../data/derived/Experiment2_lambda_F1F2_frame_level.xlsx"
)

df["F1"] = df["F1"].astype(float)
df["F2"] = df["F2"].astype(float)

X = df[["lambda1", "lambda2"]]

model_F1 = LinearRegression().fit(X, df["F1"])
model_F2 = LinearRegression().fit(X, df["F2"])

print("F1 =", model_F1.intercept_, model_F1.coef_)
print("R2(F1):", model_F1.score(X, df["F1"]))

print("F2 =", model_F2.intercept_, model_F2.coef_)
print("R2(F2):", model_F2.score(X, df["F2"]))
