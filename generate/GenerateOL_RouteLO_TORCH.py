"""
generate_moe_outputs_flat.py
============================
Generate *flattened* routing‑logit dataset directly from **Mixtral‑8x7B‑Instruct‑v0.1**.
Each token becomes a 257‑long vector → `[token_id, logits_layer0_exp0 … logits_layer31_exp7]`.
The resulting tensor of shape **(N_tokens, 257)** is saved as a single ``.pt`` file.
Option 2 (single flat tensor) ⇒ fast to load, ideal for DataLoader / MLP.
"""

import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from typing import List


print("début du script de génération des outputs")


# Configuration de l'environnement
#redefinition des variables d'environnement pour Hugging Face
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf-datasets"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM
DTYPE = torch.float16

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype= DTYPE
)
model.eval()  # Met le modèle en mode évaluation
taille = 100000
dataset_name = "HuggingFaceH4/helpful-instructions"  # Nom du dataset à utiliser
# dataset_name = "codealpaca"  # Autre exemple de dataset
#dataset_name = "flytech/python-codes-25k"  # Autre exemple de dataset
### === Chargement du dataset === ###   
dataset = load_dataset(f"{dataset_name}", split=f"train[:{taille}]", trust_remote_code=True)
print(dataset[0].keys())
print(model.config)  # Affiche la configuration du modèle


# ────────────────────────────── PROCESS ────────────────────────────
all_rows: List[List[float]] = []

for sample in tqdm(dataset, desc="Generating flat rows"):
    prompt = sample["demonstration"]
    if prompt is None:
        continue  # skip malformed sample

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_ids = inputs.input_ids[0].tolist()  # python list[int]

    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True, output_hidden_states=False)
    # router_logits: List[Tensor] length 32, each (seq_len, 8)
    router_logits = outputs.router_logits  # type: ignore

    seq_len = router_logits[0].shape[0]
    assert seq_len == len(token_ids), "Mismatch seq_len"

    # Build rows token by token
    for tk_idx, tk_id in enumerate(token_ids):
        flat_vec: List[float] = [float(tk_id)]  # first entry = token_id
        for layer_logits in router_logits:  # 32 layers
            flat_vec.extend(layer_logits[tk_idx].tolist())  # 8 values
        all_rows.append(flat_vec)

    # Free GPU / RAM asap
    del outputs, router_logits
    torch.cuda.empty_cache()

print(f"Total tokens processed: {len(all_rows):,}")

# ────────────────────────────── SAVE ───────────────────────────────
print(f"Stacking to tensor ({len(all_rows)}, 257) …")
flat_tensor = torch.tensor(all_rows, dtype=DTYPE)
print("Saving")
torch.save(flat_tensor, "OL_TORCH_100k_instr.pt")
print("Done.")
