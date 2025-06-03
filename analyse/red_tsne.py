import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Chargement des données ===
data_path = "outputs/model.output/router_logits_hidden_states_INSTRUCTIONS.pt"
data = torch.load(data_path, map_location="cpu")

print(f"{len(data)} tokens chargés")

# === Préparation des vecteurs de logits ===
X = []

for token in data:
    _, logits_layers, _ = token  # [token_id, [32 × tensor(8)], ...]
    vec = torch.cat(logits_layers).tolist()  # flatten en [256]
    X.append(vec)

print("✔️ Extraction des logits terminée.")

# === t-SNE ===
print("⏳ Lancement de t-SNE...")
tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)
print("✅ t-SNE terminé.")

# === Affichage ===
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.5)
plt.title("Projection t-SNE des router logits (32 × 8)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("figures/tsne_router_logits.png", dpi=300)
plt.show()
