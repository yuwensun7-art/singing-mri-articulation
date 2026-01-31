"""
plot_articulatory_layout.py

Generate a schematic visualization of vowel distributions across
different articulators in a shared articulatory space.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "vowel": ["a", "e", "i", "o", "u"],
    "tongue_x": [-422.76, -471.68, -522.30, -414.32, -427.89],
    "tongue_y": [-228.61, -218.70, -222.19, -228.57, -211.59],
    "palate_x": [-413.51, -415.90, -403.31, -401.14, -406.06],
    "palate_y": [-133.20, -124.08, -133.93, -138.87, -134.32],
    "tip_x": [-681.34, -653.93, -676.23, -667.16, -663.67],
    "tip_y": [-190.75, -170.51, -175.01, -179.41, -133.41],
    "dorsum_x": [-498.95, -537.65, -616.70, -545.48, -552.92],
    "dorsum_y": [-182.22, -163.11, -125.09, -179.33, -158.30],
})

offsets = {
    "tip": (-300, 0),
    "dorsum": (0, 0),
    "root": (180, -120),
    "palate": (350, 180),
}

plt.figure(figsize=(7, 7))

for _, r in df.iterrows():
    plt.scatter(r["tip_x"] + offsets["tip"][0],
                r["tip_y"] + offsets["tip"][1])
    plt.text(r["tip_x"] + offsets["tip"][0],
             r["tip_y"] + offsets["tip"][1], r["vowel"])

plt.xlabel("Shared articulatory X")
plt.ylabel("Shared articulatory Y")
plt.title("Articulator-specific vowel distributions")
plt.tight_layout()
plt.show()
