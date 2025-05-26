"""
ouvre routing_weights.pt et affiche les données
forme de routing_weights.pt
{couche_id : {"weights" : tensor[8xnbtoken], "experts" : tensor[2xnbtoken]}}
"""
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
#ouvre le fichier
data = torch.load("routing_weights.pt")


print(data[0]["experts"].shape) 
print(data[0]["experts"][0][0].item())


# Initialisation du dictionnaire de fréquence
freq = torch.zeros((32, 8))

# Affichage des experts les plus utilisés par couche
for layer in tqdm(data):
        experts = data[layer]["experts"]
        for couple in experts:
            freq[layer][couple[0].item()] += 1
            freq[layer][couple[1].item()] += 1

print(freq)
freq_norm = freq.float() / freq.sum(dim=1, keepdim=True)

def plot_expert_distribution(expert_distribution):
    # === Tracé de la heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(freq_norm.numpy(), annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

    plt.title("Fréquence normalisée d’activation des experts par couche _ dataset : helpful_instructions")
    plt.xlabel("Expert")
    plt.ylabel("Couche")
    plt.tight_layout()
    plt.savefig("figures/heatmap_experts_par_couche.png", dpi=300)
print(freq_norm)
#plot_expert_distribution(freq_norm)

