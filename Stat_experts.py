"""
ouvre routing_weights.pt et affiche les données
forme de routing_weights.pt
[couche_id : {batch_id : {"weights" : tensor[8xnbtoken], "experts" : tensor[2xnbtoken]}}]
"""
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
#ouvre le fichier
#data = torch.load("router_logits_instructions.pt") # Pour le dataset helpful_instructions
data = torch.load("router_logits_codealpaca.pt")  # Pour le dataset codealpaca

N_EXPERTS = 8                           # 8 experts par couche
pairs     = list(itertools.combinations(range(N_EXPERTS), 2))  # [(0,1), (0,2)…]
pairNid  = {p: i for i, p in enumerate(pairs)}                # (i,j) -> 0-27
L = [len(data[0][i]['weights']) for i in range(len(data[0]))]
N_TOKEN = sum(L) # Nombre de jetons traités


print("Nombre de jetons traités :", N_TOKEN)
print("le premier expert de la premiere couche pour le premier token du premier batch : ",data[0][0]["experts"][0][0].item())

# Dico de fréquence d'activation des experts par couche
def calc_freq_norm(data):
    """
    Calcule la fréquence d'activation des experts par couche.
    :param data: Dictionnaire contenant les données des experts par couche.
    :return:  Tensor de fréquence normalisée d'activation des experts par couche.
    """
    freq = torch.zeros((32, 8))
    for layer in tqdm(range(len(data))):
        for prompt_id in range(len(data[layer])):
            #print("layer :", layer, "prompt_id :", prompt_id)
            experts = data[layer][prompt_id]["experts"]
            for couple in experts:
                freq[layer][couple[0].item()] += 1
                freq[layer][couple[1].item()] += 1
    #print("fréquece d'utilisation des experts :", freq)
    freq_norm = freq.float() / freq.sum(dim=1, keepdim=True)
    return freq_norm

def plot_expert_distribution(freq_norm: torch.Tensor):
    # === Tracé de la heatmap
    # Affichage des experts les plus utilisés par couche
    plt.figure(figsize=(12, 6))
    sns.heatmap(freq_norm.numpy(), annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

    plt.title("Fréquence normalisée d’activation des experts par couche _ dataset : helpful_instructions")
    plt.xlabel("Expert")
    plt.ylabel("Couche")
    plt.tight_layout()
    plt.savefig("figures/heatmap_experts_par_couche_instructions_10000.png", dpi=300)


def couples_matrix(layer_a: int, layer_b: int, data) -> torch.Tensor:
    """
    Retourne une matrice 28×28 :
        M[p, q] = nb de token pour lesquels
                  - le couple p est activé en couche A
                  - le couple q est activé en couche B
    """
    M = torch.zeros((28, 28), dtype=torch.int32)

    for prompt_id in tqdm(range(len(data[0]))):
        experts_a = data[layer_a][prompt_id]["experts"]   # shape (nb_tokens, 2)
        experts_b = data[layer_b][prompt_id]["experts"]

        assert experts_a.shape[0] == experts_b.shape[0], \
        "Les deux couches doivent avoir le même nombre de tokens"

    

        for (e1a, e2a), (e1b, e2b) in tqdm(zip(experts_a, experts_b)):
            p = pairNid[tuple(sorted((e1a.item(), e2a.item())))]
            q = pairNid[tuple(sorted((e1b.item(), e2b.item())))]
            M[p, q] += 1

    return M


freq_norm = calc_freq_norm(data)
plot_expert_distribution(freq_norm)

mat_0_1 = couples_matrix(0, 1, data)   
print(mat_0_1)
mat_0_1 = mat_0_1.float() / mat_0_1.sum(dim=1, keepdim=True)

#plot_expert_distribution(freq_norm)

