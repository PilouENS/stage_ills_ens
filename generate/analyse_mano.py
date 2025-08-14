import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import os 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

print("Début du script")
# Configuration de l'environnement
#redefinition des variables d'environnement pour Hugging Face
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf-datasets"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16
)
model.eval()  # Met le modèle en mode évaluation

print("Config du modèle : ", model.config)

def analyze_prompt(prompt: str, strategy: str = "token_id_only", layer: int = 10):
    """
    - prompt : texte brut (ex: "The cat sat")
    - strategy : méthode de prédiction des experts ('token_id_only', 'hidden_only', 'token_id+hidden')
    - layer : couche à analyser
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  # (seq_len,)

    # Forward
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True, output_hidden_states=True)

    router_logits = outputs.router_logits  # List[layer][batch, seq_len, n_experts]
    hidden_states = outputs.hidden_states  # List[layer][batch, seq_len, dim]

    predictions = []
    mismatches = []

    for t in range(1, input_ids.size(0)):  # on commence à t=1
        context = input_ids[:t]  # tok_0 ... tok_{t-1}
        hidden_t = hidden_states[layer][0, t-1]  # hidden_{t-1, l}

        # stratégie de prédiction
        if strategy == "token_id_only":
            pred = predict_experts_from_tokens(context)  # à définir
        elif strategy == "hidden_only":
            pred = predict_experts_from_hidden(hidden_t)
        elif strategy == "token_id+hidden":
            pred = predict_experts_combined(context, hidden_t)
        else:
            raise ValueError("Unknown strategy")

        # vérité
        true_logits = router_logits[layer][0, t]  # (n_experts,)
        top2_true = torch.topk(true_logits, 2).indices.tolist()

        mismatch = sorted(pred) != sorted(top2_true)
        predictions.append((pred, top2_true))
        mismatches.append(mismatch)

    return input_ids, predictions, mismatches



def plot_expert_prediction_matrix(input_ids, predictions, save_path=None):
    """
    Input : 
    - input_ids : liste des IDs de tokens
    - predictions : liste de tuples (prédiction, vérité) pour chaque token
    - save_path : chemin pour sauvegarder la figure (optionnel)
    Affiche un tableau [token × expert] avec couleurs :
    - vert : expert correctement prédit
    - rouge : expert activé mais manqué
    - gris clair : inactif
    """
    n_tokens = len(predictions)
    n_experts = 8
    matrix = np.zeros((n_tokens, n_experts))

    for i, (pred, true) in enumerate(predictions):
        for e in true:
            if e in pred:
                matrix[i][e] = 1  # vert = bien prédit
            else:
                matrix[i][e] = -1  # rouge = expert manqué

    fig, ax = plt.subplots(figsize=(n_experts, n_tokens * 0.5))

    cmap = plt.cm.get_cmap('RdYlGn')
    ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(n_experts))
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])

    vocab = torch.load("./vocab/id_to_token.pt")
    tokens = [vocab[id] for id in input_ids.tolist()]
    #tokens = [tokenizer.decode([tid]) for tid in input_ids[1:n_tokens+1]]
    ax.set_yticks(np.arange(n_tokens))
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Experts")
    ax.set_ylabel("Tokens")
    plt.title("Prédiction des experts (vert = hit, rouge = miss)")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print("Figure sauvegardée dans :", save_path)
    else:
        plt.show()
    plt.close()
