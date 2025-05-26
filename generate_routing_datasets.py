import os

# === Configuration de l'environnement ===
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

"""
Scipt pour générer un dataset de routage à partir du modèle Mixtral-8x7B-Instruct-v0.1.
Le dataset est constitué de triplets (X, R) où X est une phrase et R est le chemin de routage correspondant.
Le chemin de routage est une liste de listes d'experts sélectionnés pour chaque token de la phrase.
Le script charge le modèle Mixtral-8x7B-Instruct-v0.1 et le tokenizer associé, puis génère des triplets (X, R) pour un nombre donné de phrases.
Le dataset est ensuite sauvegardé dans un fichier .pt.
"""

"""
Output : 
{
  "prompt": str,
  "input_ids": Tensor(seq_len),
  "routing_path": Tensor(L, seq_len, k)  # ex: (32, 128, 2)
}
"""

# === Configuration du modèle ===
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
max_seq_len = 128
num_samples = 1000# Nombre de phrases à traiter
top_k = 2  # Nombre d’experts sélectionnés par token
save_path = f"data/routing_dataset_wikitext_{num_samples}.pt"

os.makedirs("data", exist_ok=True)

# === CHARGEMENT DU DATASET ===
print("on charge le dataset")
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t.strip() for t in dataset["text"] if len(t.strip()) > 0][:num_samples]

# === TOKENIZER ET MODELE ===
print("on charge le model mixtral et le tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# === COLLECTE DES TRIPLETS ===
dataset = []
print("on genere des triplets (X, R)...")

for prompt in tqdm(texts, desc="Processing prompts"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=False,
            output_router_logits=True,
            return_dict=True
        )

    router_logits = outputs.router_logits  # Liste de (batch=1, seq_len, num_experts)
    selected_experts = []

    for layer_logits in router_logits:
        # Sélection des top-k experts pour chaque token
        topk = torch.topk(layer_logits[0], k=top_k, dim=-1).indices  # (seq_len, k)
        selected_experts.append(topk.cpu())  # Liste de tensors (L x seq_len x k)

    routing_path = torch.stack(selected_experts)  # (L, seq_len, k)

    dataset.append({
        "input_ids": inputs["input_ids"].squeeze(0).cpu(),  # on rebascule dans cpu pour torch save
        "prompt": prompt,
        "routing_path": routing_path  # (L, seq_len, k)
    })

# === SAUVEGARDE ===
torch.save(dataset, save_path)
print(f"Dataset sauvegardé dans {save_path} ({len(dataset)} exemples)")
