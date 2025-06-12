# Mixture of experts 

## Environnement de travail 
 [use_grid_5000](./use_grid_5000.md)


## Objectifs et motivations
- prédiction des experts utilisés pour un token (prédiction spatiale): pour pouvoir charger les experts en avance notamment
- prédiction des experts entre token (prédiction temporelle) : à la couche l du token t on veut savoir quels experts on utilisera à la couche l du token t+1. Ca permet de savoir quels experts on garde en mémoire
- explicabilité : le mécanisme de routage, la spécialisation ou non des experts. On aimerait mieux comprendre comment tout ça se passe.

## Ressources 
- dossier `biblio`
- [Mixture of Experts: A Smarter Way to Train AI Models (IBM)](https://www.ibm.com/think/topics/mixture-of-experts)
- [AI Expert Speculates on GPT-4 Architecture (wandb.ai)](https://wandb.ai/byyoung3/ml-news/reports/AI-Expert-Speculates-on-GPT-4-Architecture---Vmlldzo0NzA0Nzg4)

## modèles utilisés
- Mixtral 8x7B :  
    - 47B parameters (as some are shared between experts), 
    - 13B active for inference
    - 32 couches, deux experts parmi 8
    - via hugging_face/transormers
    - ~45 GB en 16bit

## Solutions pour la prédiction d'experts dans llm moe  
- (1) utilisation des gating functions des couches suivantes (prédiction spatiale)
- (2) réseau entrainé pour prédire à partir du token
- (3) corrélation/causalité entre paires d'experts

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
Attention à la taille du réseau utilisé, on ne veut pas un 'bazooka' pour tuer un moustique. Cela permet sans doute une précision intéressante mais on perd le côté explicabilité et on n'en apprends pas plus sur les experts. regarder si expertflow serieux.

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

#### Visualisation et résultats
On trace donc la matrice de co-occurence avec les deux types de normalisation. 

### Heatmap d'utilisation des experts
On trace l'utilisation des experts en fonctions des couches à partir des trajectoires. On prend ici les trajectoires qui nous intéresse, soit pour l'ensemble des token : 
![](figures/Mixtral_8x7B/heatmap_experts_helpful-instructions_10.png) *Utilisation des experts en fonctions de la couche pour le dataset Helpful-instructions*  
On observe bien que même pour un nombre restreint de d'entrée (ici 10 prompts => 448 tokens). On a une utilisation quasi-uniforme des experts à travers les couches. On retrouve ici le souhait d'équilibrer l'utilisation des experts lors de l'entrainement (notamment avec du router_noise) afin de tirer parti de l'ensemble des experts et donc de leurs poids.  
On peut aussi regarder les trajectoires d'un token donné, par exemple ici avec le token de start de chaque prompt :
![](figures/Mixtral_8x7B/heatmap_tk1_experts_helpful-instructions_10.png) *Utilisation des experts en fonctions de la couche pour le dataset Helpful-instructions pour token de start*  
On a ici une heatmap déterministe avec une seule trajectoire qlq soit le token de start pris dans le dataset. C'est un résultat rassurant car lors du forward le contexte ne contient que le passé et donc ce token (au début donc contexte "vide") a tjrs le même contexte.   
On regarde maintenant au contraire le token 28725 (',') qui est dans des contextes très différents à chaque fois.
![](figures/Mixtral_8x7B/heatmap_tk28725_experts_helpful-instructions_10.png) *Utilisation des experts en fonctions de la couche pour le dataset Helpful-instructions pour token ','*  
On a ici quelque chose de bcp plus 'flou', il n'y a pas une trajectoire qui sort du lot même si on peut observer des experts plus utilisés.   
On trace ici les statistiques avec pour seule info le token_id. Je veux regarder maintenant les trajectoires en ayant comme informartion le token_id du token qui m'intérrese mais aussi le token_id du token précédent. 
Je choisis donc un token en particulier : 28804 ('?') je regarde son utilisation des experts sans informations (je construit les trajectoires que pour ce token_id) :
![](figures/Mixtral_8x7B/heatmap_tk28804_experts_helpful-instructions_10.png) *Utilisation des experts en fonctions de la couche pour le dataset Helpful-instructions pour token '?'*  
On retrouve ici qlq chose de similaire que pour le token ',' car ce sont des token au cotexte très varié.  
On rajoute maintenant l'information du token_id précédent pour voir si on converge vers une trajectoire. 
Pour faire cella on cherche les token '?' dans data et on build sa trajectoire ssi le token_id du token précédent dans data et celui du toekn '_it' car dans ce petit dataset on a plusieurs fois l'enchainement '_it ?'. 378 et 28804


### Trajectoires 
Grâce à la matrice de co-occurence entre deux couches successives on a une information locale. On aimerait étenndre cette information sur l'ensemble des couches pour un token : analyse des trajectoires. 
On peut visualiser les trajectoires de plusieurs manières :
- réduction de la dimensions des données :
    - PCA
    - t-SNE
- Courbes 3D (E1, E2, layer) en refléchissant à dans quel ordre on range E1 et E2 
- courbes paramétrées f(E1(l), E2(l)) avec l in layer

#### Réduction de dimensions - flatten
Les algorithmes PCA et t-SNE servent à réduire la dimensions de vecteurs pour pouvoir les visualiser dans le plan et potentiellement faire de la segmentation. 
Pour utiliser ces algo il faut choisir nos vecteurs de départs. Nous avons à dispositions (récupéré depuis le modèle) pour chaque token :
- son token-id (sortie du tokenizer mixtral)
- les routeurs logits pour chaque token_id (32 couches * 8 logits)
- et donc les deux experts utilisés à chaque couche (32 couches * 2 experts_indices )
- l'embedding (4096 logits)
- hidden_vectors (32 * 4096 logits)
Pour ces deux algorithmes on veut un vecteur flatten (dim=N*1).
Pour cela on a plusieurs possibilité :
- vecteur de taille 2*32 = 64 contenant les indices des experts utilisés pour chaque couche  
Vecteur le plus petit que l'on puisse prendre qui contient l'information la plus importante (celle que l'on veut prédire). Il faut faire attention ou en tout cas refléchir à comment on ordonne E1 et E2 les deux experts de la couche l. Est-ce qu'on met tjrs l'expert le plus probable en premier : mais dcp (7, 3) != (3, 7) donc peut être pas très logique pcq pour la prédiction c'est presque la même chose pour nous. Ou alors on peut ordonner tjrs de la même manière en mettant par exemple l'indice le plus grand en premier ainsi (3,7) = (7,3). On perd icj l'information de l'expert le plus probable mais on harmonise les couples (28 couples possibles au lieux de 56). 
- vecteur de taille 32*8=256 conentnant les logits pour chaque expert pour chaque couche
- vecteur de taille 32*4=128 contenant les logits et indice des deux experts choisis pour chaque couche : mais jsp comment ranger ça

#### Réduction de dimensions - multidim
Ils existent aussi des algorithmes qui ne necessitent pas de flatten l'information en entrée (typiquement pour des images en RGB c'est mieux).
 



### Similarité des experts 
demandé papier à Pablo pour outils stat

## Calcul de la précision de la prédiction 
on run le modèle sur un dataset choisi (étudier l'effet du dataset en essayant pour différente dataset). On enregistre les experts (top-2) selectionnés ppur chaque token dans chaque couche.
On regarde à chaque fois si on a un hit (expert prédit in experts réalité). Et on a notre précision moyenne.



## Résultats



# Reunions MAJ

## semaine du 3 Juin
- papier de ELI : lien entre grammaire des tokens et trajectoires dans les routeurs (clusters par type de mot : noms, adj etc) visualisables avec tsne 
- regarder les trajectoires des tokens avec même id 
- stockage pour large dataset 
- heatmap : normalisation par ligne ou sur ensemble de la matrice ? (proba condi ou conjointe)


## semaine du 10 Juin
- demande stockage supp de 300G sur le home (en attente)
- on attends pour faire tourner sur un autre dataset dcp
- recode la génération des routers logits et hidden vectors + embbedings sans toucher à modeling
- recode des fonctions d'analyses pour s'adapter au nouveau format de generation
- visualisation des deux heatmaps
- visualisation pour 2 token_id (les deux plus présents dans mon échantillon) des experts les plus used
    - pas super marque voir pas du tout suivant le token_id et ce même 
    - **Pour le token de démarrage (token_id = 1) Instructions :**  
        ![](./figures/Mixtral_8x7B/heatmap_EXP_tk_id_1_INSTRUCTIONS_100.png)  
        *Heatmap illusatrant la fréquence d'utilisation des experts pour le token de démarrage (token_id = 1 pour) Instructions*
    - **Pour le token le plus fréquent (the) (token_id = 13) Instructions :**  
        ![](./figures/Mixtral_8x7B/heatmap_EXP_tk_id_13_INSTRUCTIONS_100.png)  
        *Heatmap illusatrant la fréquence d'utilisation des experts pour le token de démarrage (token_id = 13 pour) Instructions*

UP : figures pas bonnes donc deleted
Dans la heatmap du token d'initialisation on observe du bruit au moins dans la selection des experts de la couches 0. En effet à priori cette couche est déterministe car elle n'a pas d'autre contexte. On a vérifié et router_jitter_noise = 0.
On a créé [id_to_token](./outputs/prompt/id_to_token.pt) et [token_to_id](./outputs/prompt/token_to_id.pt) pour avoir les relations entre token_id et le vocabulaire.


- tester avec preposition particulière 
- vmap de 0 à 1 à forcer DONE : il faut retracer mais pas de ressources
- verifier couche 33 embeding
- jitter noise
- tsne : 32*8 flatten : tous en gris sauf selection de token en couleur : essayer enchainement de token (plus probable avant the +the)
write; actions dans instructions ; syntaxe vs nm variable

- ajouter info pour heatmap : prendre un token qui arrive souvent avant the et voir heatmap => essayer d'augmenter proba des traj en rajoutant de l'info

- plus tard : classificateur : obj. final

### Feuille de route :
- generate output pour trois dataset : helpful instr; code; autre à choisir. 
- analyse des token les plus used
- verifier que same(token_id) => same(embedding)
- regarder enchainement de token pour voir si ajout du contexte comme informations nous aide 
- regarder debut1 + mot1 et debut1 + mot2 si suivant la longueur de début ça nous aide à prédire la traj de mot et si c'est robuste aux variations. 



Recode PROPRE de toute la génération on ests ur que c'est good au moins. 


