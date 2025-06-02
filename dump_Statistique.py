# nouveau format : list[ n_tokens ] avec pour chaque token :
#   [ token_id : int ,
#     logits_32x8 : torch.Tensor(32,8) ,
#     hiddens_33x4096 : torch.Tensor(33,4096) ]

import torch, itertools

N_LAYERS  = 32
N_EXPERTS = 8
pairs     = list(itertools.combinations(range(N_EXPERTS), 2))
pair_idx  = {p: i for i, p in enumerate(pairs)}           # (i,j) → 0-27


# ------------------------------------------------------------------
# 1) fréquence normalisée d’activation de CHAQUE expert par couche
# ------------------------------------------------------------------
def calc_freq_norm_v2(raw_tokens):
    """
    raw_tokens : liste du nouveau format
    return      : Tensor (32,8)  où F[l,e] = fréquence normalisée
    """
    freq = torch.zeros((N_LAYERS, N_EXPERTS))

    for _, logits_32x8, _ in raw_tokens:
        # logits_32x8  : (32,8)
        top2 = logits_32x8.topk(2, dim=1).indices        # (32,2)
        for l in range(N_LAYERS):
            e1, e2 = top2[l].tolist()
            freq[l, e1] += 1
            freq[l, e2] += 1

    return freq / freq.sum(dim=1, keepdim=True)          # normalisé par couche



# ------------------------------------------------------------------
# 2) matrice de co-occurrence des couples d’experts entre 2 couches
# ------------------------------------------------------------------
def couples_matrix_v2(layer_a, layer_b, raw_tokens):
    """
    M[p,q] = nb de tokens pour lesquels
             - le couple p (ordre trié) est activé en couche A
             - le couple q      ''        ''        ''   couche B
    """
    M = torch.zeros((28, 28), dtype=torch.int32)

    for _, logits_32x8, _ in raw_tokens:
        top2 = logits_32x8.topk(2, dim=1).indices        # (32,2)
        couple_a = tuple(sorted(top2[layer_a].tolist()))
        couple_b = tuple(sorted(top2[layer_b].tolist()))
        p = pair_idx[couple_a]
        q = pair_idx[couple_b]
        M[p, q] += 1

    return M



# ------------------------------------------------------------------
# 3) trajectoires : pour CHAQUE token → liste de 32 couples d’experts
# ------------------------------------------------------------------
def trajectories_v2(raw_tokens):
    """Retourne une list[ n_tokens ][ 32 ] de tuples (e1,e2)"""
    trajs = []
    for _, logits_32x8, _ in raw_tokens:
        top2 = logits_32x8.topk(2, dim=1).indices        # (32,2)
        traj = [tuple(sorted(t.tolist())) for t in top2] # 32 couples
        trajs.append(traj)
    return trajs
