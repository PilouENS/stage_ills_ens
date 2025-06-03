import matplotlib.pyplot as plt
from Analyse_predictions_top1 import calcul_hit

# Choix des valeurs de k
k_values = range(6)

# Calcul des taux de hits pour chaque k
taux_values = [calcul_hit(k)[2] for k in k_values]  # [2] = taux_hits

import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (6, 4),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "serif"
})

# Valeurs de k à tester
k_values = range(9)

# Calcul des taux de hits
taux_values = [calcul_hit(k)[2] for k in k_values]

# Tracé
fig, ax = plt.subplots()
ax.bar(k_values, taux_values, color="firebrick", width=1.0, align='center', edgecolor="black")

# Étiquettes
ax.set_xlabel(r"$k$")
ax.set_ylabel("Taux de hits")
ax.set_title("Taux de hits en fonction de $k$")
ax.set_ylim(0, 1)
ax.grid(False)

# Affichage des valeurs sur les barres
for i, v in enumerate(taux_values):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=10)

# Sauvegarde + affichage
plt.tight_layout()
plt.savefig("trace_taux_hits.png", dpi=300)