On modifie le fichier modeling_mixtral pour en tirer ce dont on a besoin, routing paths, routings weights etc.
suivant ce qu'on veut en sortie on a fait plusieurs fichiers, il suffit de remplacer le fichier d'origine dans mixtral par le notre pour avoir les sorties.
## Versions :
# modeling_mixtral_genrate__true.py 
Permet de génerer le fichier routing_weights.pt
avec résultats sous la forme : {couche_id : {"weights" : tensor[8xnbtoken], "experts" : tensor[2xnbtoken]}}
analyse de ce fichier via Stat_experts.py
