# Mixture of experts 
il faut se mettre sur le noeud avant (voir ([use_grid_5000](./use_grid_5000.md)))
Script bash ([init_mixtral](./pilou_git/init_mixtral_node.sh)) pour mettre en place l'environnement :
- création d'un venv
- maj de pip et install des bibli py
- gestion token huggin face 
- on se place sur le noeud pour telecharger
- download transformers en editable 

Pour lancer un job sans interactive mode : submit_wait.sh

## Objectifs et motivations
- prédiction des experts utilisés pour un token (prédiction spatiale): pour pouvoir charger les experts en avance notamment
- prédiction des experts entre token (prédiction temporelle) : à la couche l du token t on veut savoir quels experts on utilisera à la couche l du token t+1. Ca permet de savoir quels experts on garde en mémoire
- explicabilité : le mécanisme de routage, la spécialisation ou non des experts. On aimerait mieux comprendre comment tout ça se passe.

## Solutions pour la prédiction d'experts dans llm moe  
- (1) utilisation des gating functions des couches suivantes (prédiction spatiale)
- (2) réseau entrainé pour prédire à partir du token
- (3) corrélation/causalité entre paires d'epxerts

### 1 - Gating function
_Prédictions horizontale (spaciale)_
On se place à la couche _k_ et on évalue le hidden vector (sortie de l'attention) avec la gatting function de la couche _k+n_ pour une prédiction à de _n_ couches dans le futur.
On regarde les prédictions top-1 et top-2. Pour top-1 on regarde si l'expert avec la meilleure prédiction se trouve bien dans la paire d'experts utilisés en réalité. Pour top-2 enregistre un hit si un deux experts prédits et utilisés et deux hit si la paire prédite et la même que la paire utilisée. 
++ : précision ok (96%, 90%) poir le top-1 et autour de 85% pour top 2 avec _n=1_ 
-- : demande de relier les couches entre elles mais temps de calcul négligeable par rapport aux calculs des experts et gain de temps ++
Pour preload les experts on le ferait techniquement juste à la couche d'avant, c'est donc ce taux de réussite qui nous interresse le plus.

Idée d'amélioration : 

### 2 - Réseau 
Ca peut être une prédiction et spatiale et temporelle 
Attention à la taille du réseau utilisé, on ne veut pas un 'bazooka' pour tuer un moustique. Cela permet sans doute une précision intéressante mais on perd le côté explicabilité et on n'en apprends 
pas plus sur les experts. regarder si expertflow serieux.

### 3 - Statistiques
On s'intéresse à la causalité ou la corrélation entre les paires d'experts utilisés entre couche puis entre token. Une piste est de trouver une loi conjointe expérimentale entre X et Y (et Z et +), X et Y 
étant les variables aléatoires des paires d'experts choisis pour deux couches consécutives (alphabet de (8 2) = 28).

to do 
- tracer la matrice de corrélation entre X et Y avec du comptage (puis étendre à Z) puis distribution jointe empirique 
- regarder les métriques pour exploiter ça (3 types de corrélation, bi-clustering)

## Méthodes 

### Matrice de co-occurence
Avec Stat_experts.py on calcule le nombre de fois que chaque couple d'experts de la couche A et utilisés par la couche B. Je veux normaliser ce résultat sur [0,1]. On peut soit normaliser la matrice par ligne, par colonne ou toute la matrice directement. 

#### Normalisation par ligne
Lorsque l'on normalise la matrice de co-occurrence (28x28) **par ligne**, on calcule une probabilité conditionnelle de la forme :
**P(couple_j à la couche L+1 | couple_i à la couche L)**
Donc on mesure la probabilité que le couple d’experts j soit activé en couche L+1, **sachant** que le couple i l’a été en couche L.
La normalisation par ligne est utile pour :
- étudier les **transitions inter-couches** dans un modèle MoE ;
- analyser le **comportement dynamique du routeur** ;
- construire des **trajectoires de routing** couche par couche.

Cette approche peut être trompeuse si on l'interprète seule.
Par exemple, un couple i **très rarement activé** peut, à chaque fois ou presque mener au même couple j.  
Dans ce cas :
- La **probabilité conditionnelle** P(j | i) sera proche de 1
- Mais la **probabilité jointe** P(i → j) = P(i) × P(j | i) sera **très faible**

> on observe un lien très fort entre i et j, mais ce chemin est **statistiquement très rare** (donc pas useful)

#### Normalisation sur toute la matrice
SI on normalise la matrice de co-occurrence (28x28) **sur l'ensemble de ses éléments**, on obtient une **probabilité jointe** :
**P(couple_i à la couche L et couple_j à la couche L+1)**
A haque case (i, j) de la matrice donne directement la **fréquence absolue** du chemin i → j dans les données.

Cette approche est utile pour :
- identifier les **transitions réellement fréquentes** dans le modèle ;
- **pondérer les liens conditionnels** avec leur fréquence d’occurrence ;
- détecter les **chemins dominants** dans la dynamique globale du routing.

La probabilité jointe intègre **à la fois la probabilité de départ (P(i)) et la conditionnelle (P(j | i))**, ce qui permet de :
- éviter de surinterpréter des chemins rares,
- **quantifier l’importance réelle** d’un chemin i → j.
Mais dcp elle ne permet pas de détecter **régularités locales** ou des règles de transition, car elle ne tient pas compte du conditionnement.  

> Un couple j peut apparaître très souvent simplement parce qu’il est populaire globalement, **sans dépendre de i**.

### Trajectoires 
Grâce à la matrice de co-occurence entre deux couches successives on à une information locale. On aimerait étenndre cette information sur l'ensemble des couches pour un token : analyse des trajectoires. 
On peut visualiser les trajectoires de plusieurs manières :
- réduction de la dimensions des données :
    - PCA
    - t-SNE
- Courbes 3D (E1, E2, layer) en refléchissant à dans quel ordre on range E1 et E2 
- courbes paramétrées f(E1(l), E2(l)) avec l in layer


#### Visualisation et résultats
On trace donc la matrice de co-occurence avec les deux types de normalisation. 

### Similarité des experts 
demandé papier à Pablo pour outils stat

## Calcul de la précision de la prédiction 
on run le modèle sur un dataset choisi (étudier l'effet du dataset en essayant pour différente dataset). On enregistre les experts (top-2) selectionnés ppur chaque token dans chaque couche.
On regarde à chaque fois si on a un hit (expert prédit in experts réalité). Et on a notre précision moyenne.



## Résultats