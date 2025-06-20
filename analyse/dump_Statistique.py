#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse des sorties Mixtral 8×7B – nouveau format
contenu dans outputs/model.output/router_logits_hidden_states_{NOM_DU_DATASET}.pt
=================================================

Format du .pt :
    [
        token_id : int,
        router_logits : List[List[float]]  # 32 couches × 8 logits
        hidden_states : List[List[float]]  # 33 couches × 4096
    ]  (un élément par token)
"""

# ──────────────────────────── Imports ────────────────────────────
import os
import itertools
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go

# ────────────────────────── Paramètres ───────────────────────────
# Chemin du fichier à analyser
DATASET_NAME = "helpful-instructions_1000"          # "INSTRUCTIONS_100", "codealpaca"
#DATA_PATH = Path(f"outputs/model.output/router_logits_hidden_states_{DATASET_NAME}.pt")
DATA_PATH = Path("pilou_git/outputs/model.output/OL_router_logitshelpful-instructions_10000.pt")
FIG_DIR = Path("figures/Mixtral_8x7B")
FIG_DIR.mkdir(parents=True, exist_ok=True)  # Crée le dossier s'il n'existe pas


N_LAYERS   = 32          # couches « transformer »
N_EXPERTS  = 8           # experts par couche
TOPK       = 2           # nb d’experts que l’on retient (top-k)

# ──────────────────────── Chargement data ────────────────────────
data = torch.load(DATA_PATH)
"""
data = np.load(DATA_PATH, allow_pickle=True)
print("to list")
data = data.tolist()  # Pour obtenir une vraie list[list[...]]
print(len(data), "tokens chargés depuis", DATA_PATH)
for i in tqdm(range(len(data)), desc="Conversion en tensor"):
    data[i][1] = [torch.tensor(logits) for logits in data[i][1]]
"""


# ──────────────────────── Pré-calcul commun ──────────────────────
pairs      = list(itertools.combinations(range(N_EXPERTS), 2))   # [(0,1)…]
pair2idx   = {p: i for i, p in enumerate(pairs)}                 # (i,j)->0-27
IDX2PAIR   = {v: k for k, v in pair2idx.items()}                 # 0-27->(i,j)   

# Pour chaque token : [ (e1, e2) ] × 32
def build_trajectories():
    trajs = []
    for _, layers, _ in tqdm(data):
        traj = []
        for layer_logits in layers:              # 8 logits
            top2 = torch.topk(layer_logits, TOPK).indices.tolist()
            traj.append(tuple(sorted(top2)))
        trajs.append(traj)
    return trajs

#TRAJ = build_trajectories()     # pré-calcul pour être réutilisé
#assert len(TRAJ) == len(data)
TRAJ = []
# ─────────────────── 1) Fréquence d’activation ───────────────────
def calc_freq_norm(TRAJ=TRAJ):
    """freq[layer, expert] = % d’activation sur tous les tokens."""
    freq = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long)
    for traj in tqdm(TRAJ, desc="Calcul fréquence d'activation"):
        for layer, (e1, e2) in enumerate(traj):
            freq[layer, e1] += 1
            freq[layer, e2] += 1
    return freq.float() / freq.sum(dim=1, keepdim=True)

def plot_expert_distribution(name, TRAJ=TRAJ): 
    """Trace la heatmap de la fréquence d’activation
     des experts en fonction des couches."""
    freq_norm = calc_freq_norm(TRAJ)
    plt.figure(figsize=(12, 6))
    sns.heatmap(freq_norm.T, annot=False, cmap="YlGnBu", vmax=0.5)
    plt.title(f"Fréquence normalisée d’activation (dataset : {DATASET_NAME}), nb tokens : {len(TRAJ)}")
    plt.ylabel("Expert")
    plt.xlabel("Couche")
    plt.tight_layout()
    save_path = FIG_DIR / f"{name}"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Heatmap experts sauvegardée → {save_path}")

# ─────────────── 2) Matrice de co-occurrence couples ─────────────
def couples_matrix(layer_a: int, layer_b: int, TRAJ=TRAJ):
    """
    M[pa, pb] = #tokens avec le couple pa (layer_a) ET pb (layer_b)
    pa/pb codés 0-27 via pair2idx.
    """
    M = torch.zeros((len(pairs), len(pairs)), dtype=torch.long)
    for traj in TRAJ:
        pa = pair2idx[traj[layer_a]]
        pb = pair2idx[traj[layer_b]]
        M[pa, pb] += 1
    return M

def trace_couples_matrix(layer_a: int, layer_b: int, TRAJ=TRAJ, save_path=None):
    M = couples_matrix(layer_a, layer_b, TRAJ=TRAJ)
    row_norm = torch.zeros_like(M, dtype=torch.float)
    for ligne_id in range(len(M)):
        row_norm[ligne_id] = M[ligne_id] / (M[ligne_id].sum() if M[ligne_id].sum()>0 else 1)
    joint_norm = M.float() / M.sum()

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(row_norm, cmap="YlOrBr", vmax=row_norm.max(),
                xticklabels=pairs, yticklabels=pairs, ax=axs[0])
    axs[0].set_title(f"P(couple B | couple A) — {layer_a}→{layer_b}")
    axs[0].set_xlabel(f"Couple couche {layer_b}")
    axs[0].set_ylabel(f"Couple couche {layer_a}")
    axs[0].tick_params(axis='x', rotation=45)

    sns.heatmap(joint_norm, cmap="YlGnBu", vmax=joint_norm.max(),
                xticklabels=pairs, yticklabels=False, ax=axs[1])
    axs[1].set_title(f"P(couple A ∧ couple B) — {layer_a} & {layer_b}")
    axs[1].set_xlabel(f"Couple couche {layer_b}")
    axs[1].tick_params(axis='x', rotation=45)
    if save_path is None:
        save_path = FIG_DIR / f"couples_matrix_{layer_a}_{layer_b}_{DATASET_NAME}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Matrice couples sauvegardée → {save_path}")

# ─────────────── 3) Trajectoires 3D (Plotly) ───────────────
def plot_all_trajectories_3d(TRAJ=TRAJ, max_tokens=500, save_path=None):
    fig = go.Figure()
    for traj in TRAJ[:max_tokens]:
        x = [max(e) for e in traj]   # expert max
        y = [min(e) for e in traj]   # expert min
        z = list(range(N_LAYERS))    # couches
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(width=2),
            opacity=0.15,
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Expert max",
            yaxis_title="Expert min",
            zaxis_title="Couche",
            aspectratio=dict(x=1, y=1, z=7)  # étire axe z
        ),
        title=f"Trajectoires des experts ({min(len(TRAJ), max_tokens)} tokens)",
        width=900, height=700
    )
    fig.write_html(str(save_path))
    print(f"Figure 3D interactive → {save_path}")

# ────────────────────────── Fréquence token_id ──────────────────────────
def analyse_token_id_distribution():
    token_ids = [tok[0] for tok in data]
    freqs = torch.bincount(torch.tensor(token_ids))
    freqs = freqs.float() / freqs.sum()  # Normalisation
    return freqs

def build_TRAJ_fromTKID(tk_max=13):
    trajs = []
    
    for tok_id, layers, _ in data:
        traj = []
        if tok_id == tk_max:
            for layer_logits in layers:              # 8 logits
                top2 = torch.topk(layer_logits, TOPK).indices.tolist()
                traj.append(tuple(sorted(top2)))
            trajs.append(traj)
    print(f"lg de trajs pour token = {tk_max}:", len(trajs))
    return trajs

def build_TRAJ_fromDATA_ID(data_id):
    trajs = []
    
    for id in data_id:
        traj = []
        _, layers, _ = data[id]
        for layer_logits in layers:              # 8 logits
            top2 = torch.topk(layer_logits, TOPK).indices.tolist()
            traj.append(tuple(sorted(top2)))
        trajs.append(traj)
    print(f"lg de trajs pour id = {data_id} : ", len(trajs))
    return trajs


### ----------- build traj 
def rang_INdata_fromtk_id(tk=13):
    "retourne les indices des tokens dans data pour le token_id donné"
    id = []
    for i in range(len(data)):
        if data[i][0] == tk:
            id.append(i)
    print(f"lg de trajs pour token = {tk}:", len(id))
    return id

def rang_INdata_from_2TK(tkPREC, tkCIBLE):
    "retourne les indices des tokens dans data pour le token cible ssi il est precede par le token precedent"
    id = []
    for i in range(len(data)):
        if data[i][0] == tkCIBLE and data[i-1][0] == tkPREC:
            id.append(i)
    print(f"token précédent {tkPREC} et cible {tkCIBLE} : lg de trajs :", len(id))
    return id

def calcul_hit_miss(TRAJ=TRAJ):
    """ Calculer la hit rate et la miss rate
    a partir des frequences d'activation si on prédit que les experts utilisés sont les plus actifs
    et on calcule la hit rate et la miss rate """
    freq_norm = calc_freq_norm(TRAJ=TRAJ)
    hits, misses, total = 0, 0, 0
    prediction = []
    for layer in freq_norm:
        prediction.append(torch.topk(layer, 2)[1].tolist())

    for traj in TRAJ:
        for layer in range(len(traj)):
            for top_experts in traj[layer]:
                if top_experts in prediction[layer]:
                    hits += 1
                else:
                    misses += 1
                total += 1

    hit_rate = hits / total if total > 0 else 0
    miss_rate = misses / total if total > 0 else 0
    return hit_rate, miss_rate
        
            

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0
    miss_rate = misses / total if total > 0 else 0

    print(f"Hit rate: {hit_rate:.2f}, Miss rate: {miss_rate:.2f}")

calcul_hit_miss()





breakpoint()

id_i_can = rang_INdata_from_2TK(315, 541)
Traj_i_can = build_TRAJ_fromDATA_ID(id_i_can)
plot_expert_distribution("heatmap_helpinst10000_i_can.png", TRAJ=Traj_i_can)
id_can = rang_INdata_fromtk_id(541)
Traj_can = build_TRAJ_fromDATA_ID(id_can)
plot_expert_distribution("heatmap_helpinst10000_can.png", TRAJ=Traj_can)

# ───────────────────────────── Main ──────────────────────────────
"""
if __name__ == "__main__":
    
    print("→ Calcul fréquence d’activation…")
    freq_norm = calc_freq_norm()
    plot_expert_distribution(freq_norm)
    
    # Exemple : co-occurrence couches 0 et 1
    #trace_couples_matrix(0, 1)


    freqs = analyse_token_id_distribution()
    print("id max :", freqs.argmax().item(), "avec fréquence :", freqs.max().item())
    
    #Trajectoires 3D (sous-échantillon pour la lisibilité)
    #plot_all_trajectories_3d(max_tokens=1000,save_path=FIG_DIR / f"trajectories_3d_{DATASET_NAME}.html")
    
    # Trajectoires pour le token_id 13
    plot_expert_distribution(TRAJ=TRAJ_tkMAX)
    
    
    trace_couples_matrix(0, 1, TRAJ=TRAJ_tkMAX, 
                         save_path=FIG_DIR / f"couples_matrix_tkMAX_0_1_{DATASET_NAME}.png")
"""
