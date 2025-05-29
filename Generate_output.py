"""
script to genreate outputs from the Mixtral-8x7B-Instruct-v0.1 model using output_router_logits and output_hidden_states
on a pas besoin de mofier modeling_mixtral.py
plusieurs datasets sont disponibles pour tester le script
save les outputs dans un fichier .pt 
    de la forme (router_logits, hidden_states, token_ids)
        data[0][layer][token number][poids x8]
        data[1][embedding+layer][token number][hidden_states x 4096]
        data[2][token_id]
"""

import os
import torch
from datasets import load_dataset
from tqdm import tqdm

# Configuration de l'environnement
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16, 
    output_hidden_states=True, 
    output_router_logits=True
)
model.eval()  # Met le modèle en mode évaluation

### === Chargement du dataset === ###   
#dataset = load_dataset("HuggingFaceH4/helpful_instructions", split="train[:1000]")
dataset = load_dataset("HuggingFaceH4/testing_codealpaca_small", split="train[:3]", trust_remote_code=True)
#dataset = load_dataset('flytech/python-codes-25k', split="train[:3]", trust_remote_code=True)
print(dataset[0].keys())
token_ids = []  # Liste pour stocker les IDs des tokens
router_logits = []  # Liste pour stocker les logits des routeurs
hidden_states = []  # Liste pour stocker les états cachés
for sample in tqdm(dataset):  # progress bar utile pour les longs datasets
    #prompt = sample["prompt"] #helpful_instructions
    prompt = sample.get("prompt", "").strip()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    token_ids.extend(inputs.input_ids[0].tolist())  # On stocke les IDs des tokens
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    router_logits_TK = outputs.router_logits  
    hidden_states_TK = outputs.hidden_states  
    for layer in range(len(router_logits_TK)):
        # On stocke les logits des routeurs et les états cachés pour chaque couche
        if layer >= len(router_logits):
            router_logits.append([])
            hidden_states.append([])
        router_logits[layer].append(router_logits_TK[layer].cpu())
        hidden_states[layer].append(hidden_states_TK[layer].cpu())
        # on concatene tous les tensors de logits et d'états cachés

# on concatene les tensors de logits et d'états cachés  
for layer in range(len(router_logits)):    
    router_logits[layer] = torch.cat(router_logits[layer], dim=0)
    hidden_states[layer] = torch.cat(hidden_states[layer], dim=0)


#on enregistre les resultats dans un fichier
torch.save((router_logits, hidden_states,token_ids), "model_output_test.pt")

