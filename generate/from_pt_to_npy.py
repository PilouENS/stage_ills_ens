import torch
import numpy as np


def convert_pt_to_npy(input_path="pilou_git/outputs/model.output/OL_router_logitshelpful-instructions_10000.pt", output_path="OL_router_logitshelpful-instructions_10000.npy"):
    print(f"Chargement du fichier .pt depuis {input_path}")
    data = torch.load(input_path, map_location="cpu")

    # Vérification et conversion : tensors -> lists
    for i in range(len(data)):
        data[i][1] = [tensor.tolist() for tensor in data[i][1]]

    print(f"Sauvegarde au format .npy vers {output_path}")
    np.save(output_path, data)
    print("Conversion terminée.")

convert_pt_to_npy()
