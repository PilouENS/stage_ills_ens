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
    torch_dtype=torch.float16, 
    output_hidden_states=True, 
    output_router_logits=True
)
model.eval()  # Met le modèle en mode évaluation

### === Chargement du dataset === ###   
#dataset = load_dataset("HuggingFaceH4/helpful_instructions", split="train[:10000]", trust_remote_code=True)
#dataset = load_dataset("HuggingFaceH4/testing_codealpaca_small", split="train[:2]", trust_remote_code=True)
dataset = load_dataset('flytech/python-codes-25k', split="train[:10000]", trust_remote_code=True)
print(dataset[0].keys())


token_ids = []  # Liste pour stocker les IDs des tokens
data = []
model.eval()
for sample in tqdm(dataset): 
    #prompt = sample["prompt"] #helpful_instructions
    #prompt = sample.get("prompt", "").strip() #codealpaca
    prompt = sample["output"]  # python-codes-25k
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    token_ids.extend(inputs.input_ids[0].tolist())  # On stocke les IDs des tokens
    #print("prompt :", prompt)
    #print("nb de tokens :", len(inputs.input_ids[0]))
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
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
            #data_new[tk][2].append(hidden_states_PROMPT[l][0][tk].to("cpu"))  # On utilise [0] pour accéder à la première dimension des états cachés
        data_new[tk][2].append(hidden_states_PROMPT[0][0][tk].to("cpu")) #embedding
    for elm in data_new:
        data.append(elm)  # On ajoute les données formatées pour chaque token à la liste data

    del outputs, router_logits_PROMPT, hidden_states_PROMPT
    torch.cuda.empty_cache()

    # data de la forme [token_id, [couche][x8], [embedding+couche][x4096]]


torch.save(data, "../outputs/model.output/router_logits_hidden_states_INSTRUCTIONS.pt")
print("finito la génération")

