# Analyse des activations d'experts (Mixture of Experts - MoE)

## `Analyse_predictions_top1.py`
Analyse la capacité à prédire correctement **l’expert principal (top-1)** activé à chaque couche. Sert à évaluer la précision d’un modèle de prédiction du routing.

## `Analyse_predictions_top2.py`
Version top-2 : vérifie si **l’un des deux experts** activés a été correctement prédit. Moins stricte, mais plus informative sur le potentiel de prédiction.

## `Stat_experts_modeling.py`
Script principal de **statistiques descriptives** pour data format modeling :  
- Fréquences d’activation des experts par couche  
- Matrices de co-occurence entre couches  
- Trajectoires inter-couches  
- Heatmaps et visualisations
A lancer sur les sorties de generation_modeling !

### `dump_Statistique.py`
Script principal de **statistiques descriptives** pour data format dump (output du model directement) :  
- Fréquences d’activation des experts par couche  
- Matrices de co-occurence entre couches  
- Trajectoires inter-couches  
- Heatmaps et visualisations
A lancer sur les sorties de generation_output !

### `trace_predict_accuracy(layer).py`
Trace la **précision de prédiction** (top-1 ou top-2) pour une ou plusieurs couches données. Sert à identifier les couches plus ou moins prévisibles.

### `trace_taux_hits(k).py`
Calcule et trace le taux de “hit” pour différents **k** (dans le top-k). 

