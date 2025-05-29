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

data_name = "instructions" #"codealpaca"  # Nom du dataset utilisé
data = torch.load(f"router_logits_{data_name}.pt", map_location="cuda:0")  # Pour le dataset codealpaca


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

    plt.title(f"Fréquence normalisée d’activation des experts par couche _ dataset : {data_name}")
    plt.xlabel("Expert")
    plt.ylabel("Couche")
    plt.tight_layout()
    plt.savefig(f"figures/heatmap_experts_par_couche_{data_name}.png", dpi=300)


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


def trace_couples_matrix(layer_a, layer_b, data):
    M = couples_matrix(layer_a, layer_b, data)
    row_norm = M.float() / M.sum(dim=1, keepdim=True)
    joint_norm = M.float() / M.sum()
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(row_norm, cmap="YlOrBr", vmin=0, vmax=row_norm.max(),
                xticklabels=pairs, yticklabels=pairs, ax=axs[0])
    axs[0].set_title(f"Proba conditionnelle — couche {layer_a} → {layer_b} — {data_name}")
    axs[0].set_xlabel(f"Couple en couche {layer_b}")
    axs[0].set_ylabel(f"Couple en couche {layer_a}")
    axs[0].tick_params(axis='x', rotation=45)

    sns.heatmap(joint_norm, cmap="YlGnBu", vmin=0, vmax=joint_norm.max(),
                xticklabels=pairs, yticklabels=False, ax=axs[1])
    axs[1].set_title(f"Proba jointe — couche {layer_a} ∧ {layer_b} — {data_name}")
    axs[1].set_xlabel(f"Couple en couche {layer_b}")
    axs[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"figures/couples_matrix_{layer_a}_{layer_b}_{data_name}.png", dpi=300)
    plt.close()

#freq_norm = calc_freq_norm(data)
#plot_expert_distribution(freq_norm)

#mat_0_1 = couples_matrix(0, 1, data)   
#print(mat_0_1)
#mat_0_1 = mat_0_1.float() / mat_0_1.sum(dim=1, keepdim=True)
#trace_couples_matrix(2, 3, data)  # Tracer la matrice des couples pour les couches 0 et 1



#plot_expert_distribution(freq_norm)

