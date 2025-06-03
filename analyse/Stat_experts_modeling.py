"""
ouvre routing_weights.pt (poitns de MODELING modifiés) et affiche les données
forme de routing_weights.pt
[couche_id : {batch_id : {"weights" : tensor[8xnbtoken], "experts" : tensor[2xnbtoken]}}]
"""
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

### DATA LOADER ###
#ouvre le fichier
#data = torch.load("router_logits_instructions.pt") # Pour le dataset helpful_instructions
data_name = "instructions" #"codealpaca"  # Nom du dataset utilisé
data = torch.load(f"outputs/modeling modifie/router_logits_{data_name}.pt", map_location="cuda:0")  # Pour le dataset codealpaca




### TESTE et INFOS ###
N_EXPERTS = 8                           # 8 experts par couche
pairs     = list(itertools.combinations(range(N_EXPERTS), 2))  # [(0,1), (0,2)…]
pairNid  = {p: i for i, p in enumerate(pairs)}                # (i,j) -> 0-27
L = [len(data[0][i]['weights']) for i in range(len(data[0]))]
N_TOKEN = sum(L) # Nombre de jetons traités
print("Nombre de jetons traités :", N_TOKEN)
print("le premier expert de la premiere couche pour le premier token du premier batch : ",data[0][0]["experts"][0][0].item())





### FREQUENCE ET COUPLES DES EXPERTS PAR COUCHE ###
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





### MATRICES CO OCCURRENCE DES COUPLES D'EXPERTS ###
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
        for (e1a, e2a), (e1b, e2b) in zip(experts_a, experts_b):
            p = pairNid[tuple(sorted((e1a.item(), e2a.item())))]
            q = pairNid[tuple(sorted((e1b.item(), e2b.item())))]
            M[p, q] += 1

    return M
def trace_couples_matrix(layer_a, layer_b, data):
    """
    Trace les deux heatmaps :
    - Proba conditionnelle : P(couple en couche B | couple en couche A)
    - Proba jointe : P(couple en couche B ∧ couple en couche A)
    Inputs :
    :param layer_a: Couche A (int)
    :param layer_b: Couche B (int)
    :param data: Données des experts par couche

    """
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
    plt.savefig(f"figures/2couples_matrix_{layer_a}_{layer_b}_{data_name}.png", dpi=300)
    plt.close()





### Trajectoires des experts dans l'espace 3D ###
def trajectoires ():
    trajectories = []  # Liste pour stocker les trajectoires des experts
    N_PROMPTS = len(data[0])  # Nombre de prompts
    N_LAYERS = 32      # Nombre de couches
    for prompt_id in range(N_PROMPTS):
        nb_tokens = data[0][prompt_id]["experts"].shape[0]
        for token_id in range(nb_tokens):
            traj = []
            for layer in range(N_LAYERS):
                e1, e2 = data[layer][prompt_id]["experts"][token_id]
                traj.append((e1.item(), e2.item()))
            trajectories.append(traj)
    return trajectories
def plot_all_trajectories_3d(max_tokens, save_path=None):
    """
    trajectories : List[List[Tuple[int, int]]]  # [n_tokens][n_layers]
    max_tokens : nombre max de trajectoires à tracer pour lisibilité
    """
    trajectories = trajectoires()  # Récupère les trajectoires des experts
    print(f"Nombre de trajectoires : {len(trajectories)}")
    fig = go.Figure()

    for i, traj in enumerate(trajectories[:max_tokens]):
        x = [max(e[0], e[1]) for e in traj]  # expert 1
        y = [min(e[0], e[1]) for e in traj]  # expert 2
        z = list(range(len(traj)))  # couches
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(width=2),
            opacity=0.2,
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Expert 1",
            yaxis_title="Expert 2",
            zaxis_title="Couche",
            aspectratio=dict(x=1, y=1, z=3) 
        ),
        title=f"Trajectoires des experts ({min(len(trajectories), max_tokens)} tokens)",
        width=900,
        height=700
        

    )
    fig.write_html(save_path)
    print(f"Figure interactive enregistrée dans : {save_path}")

### RUN EXPERIMENTS ###


plot_all_trajectories_3d(max_tokens=2, save_path="figures/trajectories_3d_all.html")

#freq_norm = calc_freq_norm(data)
#plot_expert_distribution(freq_norm)

#mat_0_1 = couples_matrix(0, 1, data)   
#print(mat_0_1)
#mat_0_1 = mat_0_1.float() / mat_0_1.sum(dim=1, keepdim=True)
#trace_couples_matrix(0, 1, data)  # Tracer la matrice des couples pour les couches 0 et 1



#plot_expert_distribution(freq_norm)

