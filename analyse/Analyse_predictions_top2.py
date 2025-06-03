"""
Script pour ouvrir un .pth contenant les deux dico predictions et realite 
et regarde la correlation entre nos predictions et la realite
"""
import torch
import sys

step_prediction = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# Charger le fichier


def calcul_hit(k=step_prediction):
    hit = 0
    nb_total = 0
    data = torch.load(f"prediction_realite_k={k}.pt")  # chemin à adapter
    #recuperer les dico
    prediction = data["prediction"]  # dict ou tensor
    realite = data["realite"]

    # Pour chaque couche
    for layer in prediction:
        for token in range(len(prediction[layer])):
            for expert in range(len(prediction[layer][token])):
                #print('couche : ', layer)
                #print('token : ', token)
                #print("prediction : ", prediction[layer][token][expert], "realite : ", realite[layer][token])
                # Si l'expert est le même que dans la réalité
                if prediction[layer][token][expert] in realite[layer][token]:
                    hit += 1
                    #print("hit")
                nb_total += 1

    taux_hits = hit / nb_total
    return hit, nb_total, taux_hits

# Afficher le résultat
if __name__ == "__main__":
    # Calculer le taux de hits
    hit, nb_total, taux_hits = calcul_hit()
    print("Prediction pour la couche n-: ", step_prediction)
    print("Nombre de hits :", hit)
    print("Nombre total d'experts :", nb_total)
    print("Taux de hit :", taux_hits)
    print("N*(L-1)*2 = ", 12*31*2)


