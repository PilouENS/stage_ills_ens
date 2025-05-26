import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

# Mise en forme matplotlib
mpl.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "serif"
})


def calcul_hit_par_couche(l, k):
    """
    Calcule le taux de hit pour une couche donnée (l -> prédiction de l+k)
    """
    try:
        data = torch.load(f"prompt_instructions/prediction_realite_k={k}.pt")
        prediction = data["prediction"]
        realite = data["realite"]

        if l not in prediction or (l + k) not in realite:
            return None  # Données manquantes

        pred_layer = prediction[l]
        real_layer = realite[l]

        hit = 0
        total = 0
        for token in range(len(pred_layer)):
            pred_expert = pred_layer[token][0]  # top-1 expert
            true_experts = real_layer[token]
            if pred_expert in true_experts:
                hit += 1
            total += 1

        return hit / total if total > 0 else None

    except Exception as e:
        print(f"Erreur à la couche {l} (k={k}) : {e}")
        return None


# Liste des couches à tester
layers = list(range(1, 33))

taux1 = [calcul_hit_par_couche(l, k=1) for l in layers]
taux2 = [calcul_hit_par_couche(l, k=2) for l in layers]
taux3 = [calcul_hit_par_couche(l, k=3) for l in layers]
taux4 = [calcul_hit_par_couche(l, k=4) for l in layers]

#calcul des moyennes associees
moyenne1 = sum(x for x in taux1 if x is not None) / sum(1 for x in taux1 if x is not None)
moyenne2 = sum(x for x in taux2 if x is not None) / sum(1 for x in taux2 if x is not None)
moyenne3 = sum(x for x in taux3 if x is not None) / sum(1 for x in taux3 if x is not None)
moyenne4 = sum(x for x in taux4 if x is not None) / sum(1 for x in taux4 if x is not None)

# Tracé
fig, ax = plt.subplots()
ax.plot(layers, taux1, marker='o', label=f"Next 1 (moy={moyenne1:.3f})", color='steelblue')
ax.plot(layers, taux2, marker='s', label=f"Next 2 (moy={moyenne2:.3f})", color='firebrick')
ax.plot(layers, taux3, marker='^', label=f"Next 3 (moy={moyenne3:.3f})", color='forestgreen')
ax.plot(layers, taux4, marker='x', label=f"Next 4 (moy={moyenne4:.3f})", color='darkorange')

ax.set_xlabel("Index de la couche courante $l$")
ax.set_ylabel("Précision de prédiction top-1 pour $l+k$")
ax.set_title("Précision de prédiction des experts en fonction de la couche")
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.savefig("precision_par_couche_next1_next2.png", dpi=300)
plt.show()
