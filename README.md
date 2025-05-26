# Mixture of experts 
il faut se mettre sur le noeud avant (voir ([use_grid_5000](./use_grid_5000.md)))
Script bash ([init_mixtral](./pilou_git/init_mixtral_node.sh)) pour mettre en place l'environnement :
- création d'un venv
- maj de pip et install des bibli py
- gestion token huggin face 
- on se place sur le noeud pour telecharger
- download transformers en editable 


## Solutions pour la prédiction d'experts dans llm moe  
- (1) utilisation des gating functions des couches suivantes
- (2) réseau entrainé pour prédire à partir du token
- (3) corrélation/causalité entre paires d'epxerts

### 1 - Gating function
_Prédictions horizontale (spaciale)_
On se place à la couche _k_ et on évalue le hidden vector (sortie de l'attention) avec la gatting function de la couche _k+n_ pour une prédiction à de _n_ couches dans le futur.
++ : précision ok (96%, 90%) poir le top-1 et autour de 85% pour top 2 avec _n=1_ 
-- : demande de relier les couches entre elles mais temps de calcul négligeable par rapport aux calculs des experts et gain de temps ++

### 2 - Réseau 
Ca peut être un eprédiction et spatiale et temporelle 
Attention à la taille du réseau utilisé, on ne veut pas un 'bazooka' pour tuer un moustique. Cela permet sans doute une précision intéressante mais on perd le côté explicabilité et on n'en apprends 
pas plus sur les experts. regarder si expertflow serieux

### 3 - Statistiques
On s'intéresse à la causalité ou la corrélation entre les paires d'experts utilisés entre couche puis entre token. Une piste est de trouver une loi conjointe expérimentale entre X et Y (et Z et +), X et Y 
étant les variables aléatoires des paires d'experts choisis pour deux couches consécutives (alphabet de (8 2) = 28).

to do 
- tracer la matrice de corrélation entre X et Y avec du comptage (puis étendre à Z) puis distribution jointe empirique 
- regarder les métriques pour exploiter ça (3 types de corrélation, bi-clustering)

## Calcul de la précision de la prédiction 
on run le modèle sur un dataset choisi (étudier l'effet du dataset en essayant pour différente dataset). On enregistre les experts (top-2) selectionnés ppur chaque token dans chaque couche.
On regarde a chaque fois si on a un hit (expert prédit in experts réalité). Et on a notre précision moyenne.

## Résultats
