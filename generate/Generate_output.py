"""
script to genreate outputs from the Mixtral-8x7B-Instruct-v0.1 model using output_router_logits and output_hidden_states
on a pas besoin de mofier modeling_mixtral.py
plusieurs datasets sont disponibles pour tester le script
save les outputs dans un fichier .pt 
    de la forme (token_ids, router_logits, hidden_states)
        data[0][token_id]
        data[1][layer][token number][poids x8]
        data[2][embedding+layer][token number][hidden_states x 4096]
        
"""

import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

print("début du script de génération des outputs")


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
taille = 1000
dataset_name = "HuggingFaceH4/helpful-instructions"  # Nom du dataset à utiliser
# dataset_name = "codealpaca"  # Autre exemple de dataset
# dataset_name = "python-codes-25k"  # Autre exemple de dataset
### === Chargement du dataset === ###   
dataset = load_dataset(f"{dataset_name}", split=f"train[:{taille}]", trust_remote_code=True)
print(dataset[0].keys())
print(model.config)  # Affiche la configuration du modèle

  # Liste pour stocker les IDs des tokens
entree = []
data = []
#affiche router_jitter_noise
print(f"router_jitter_noise : {model.config.router_jitter_noise}")

for sample in tqdm(dataset): 
    prompt = sample["demonstration"] #helpful_instructions
    #prompt = sample.get("prompt", "").strip() #codealpaca
    #prompt = sample["output"]  # python-codes-25k
    #print("prompt:", prompt, "\n")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_ids = inputs.input_ids[0].tolist()  # On stocke les IDs des tokens
    entree.append([prompt, token_ids])  # On stocke le prompt pour référence
    #print("token_ids:", token_ids, "\n")
    #print("input_ids:", inputs.input_ids, "\n")
    #print("\n\n")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_router_logits=True)
    router_logits_PROMPT = outputs.router_logits  #[couche][token number][poids x8] = 32xNB_tokenx8
    hidden_states_PROMPT = outputs.hidden_states  #[embedding+layer][token number][hidden_states x 4096] = 33xNB_tokenx4096
    
    data_new = []  # Liste pour stocker les données formatées pour chaque token
    for tk in range(len(router_logits_PROMPT[0])):
        data_new.append([token_ids[tk], [], []])  # Initialisation de data_new pour chaque token

    # On ajoute les IDs des tokens à la liste data
    # On stocke les logits des routeurs et les états cachés pour chaque couche
    for l in range(len(router_logits_PROMPT)):
        for tk in range(len(router_logits_PROMPT[l])):
            data_new[tk][1].append(router_logits_PROMPT[l][tk].to("cpu"))
            data_new[tk][2].append(hidden_states_PROMPT[l][0][tk].to("cpu"))  # On utilise [0] pour accéder à la première dimension des états cachés
    
    for tk in range(len(router_logits_PROMPT[l])):
        data_new[tk][2].append(hidden_states_PROMPT[l+1][0][tk].to("cpu")) #embedding

    for elm in data_new:
        data.append(elm)  # On ajoute les données formatées pour chaque token à la liste data

    del outputs, router_logits_PROMPT, hidden_states_PROMPT
    torch.cuda.empty_cache()

    # data de la forme [token_id, [couche][x8], [embedding+couche][x4096]]

output_path = Path(f"router_logits_hidden_states_{dataset_name}_{taille}.pt")
output_path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier si besoin

print("saving...")
torch.save(data, output_path)  # Sauvegarde les données dans un fichier .pt

"""
with open(f"prompts_{dataset_name}_{taille}.txt", "w") as f:
    f.write(f"Dataset: {dataset_name}, Taille: {taille}\n\n")
    f.write(str(model.config) + "\n\n")  # Écrit la configuration du modèle dans le fichier
    for pro_id in entree:
        f.write(str(pro_id[0])+ "\n")  # prompt
        f.write(str(pro_id[1])+ "\n")  # token_ids
        f.write("\n\n") # Séparateur entre les entrées
"""

print(f"finito la génération, save dans router_logits_hidden_states_{dataset_name}_{taille}.pt et prompts_{dataset_name}_{taille}.pt")

