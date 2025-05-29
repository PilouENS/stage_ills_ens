\section{Mixture of Experts}

Avant de commencer, il faut se connecter au nœud de calcul (voir \texttt{use\_grid\_5000.md}).
Le script Bash \texttt{init\_mixtral\_node.sh} (dans \texttt{./pilou\_git/}) permet de configurer l'environnement automatiquement :
\begin{itemize}
\item création d'un environnement virtuel (venv),
\item mise à jour de \texttt{pip} et installation des bibliothèques Python,
\item gestion du token Hugging Face,
\item déplacement sur le nœud pour le téléchargement,
\item clonage de \texttt{transformers} en mode editable.
\end{itemize}

\section{Solutions pour la prédiction d'experts dans les LLM MoE}

\begin{enumerate}
\item Utilisation des fonctions de gating des couches suivantes
\item Réseau entraîné pour prédire à partir du token
\item Analyse de la corrélation ou causalité entre paires d'experts
\end{enumerate}

\subsection{Utilisation des fonctions de gating}
\textit{Prédiction horizontale (spatiale)} : on se place à la couche \$k\$ et on applique la fonction de gating de la couche \$k+n\$ sur le vecteur caché \$h\_k\$ (sortie de l'attention) pour prédire les experts \$n\$ couches plus loin.

\textbf{Avantages} : bonne précision (96% top-1, 90% top-2 pour \$n=1\$).

\textbf{Inconvénients} : demande de relier les couches entre elles. Cependant, le coût de calcul reste négligeable par rapport à l'évaluation des experts eux-mêmes, et le gain de temps est important.

\subsection{Réseau prédicteur}
On peut entraîner un petit réseau (MLP, CNN...) pour prédire les experts activés à partir du token et/ou des activations passées.

Cela permet une prédiction à la fois spatiale et temporelle.\newline
\textbf{Attention} : il ne faut pas utiliser un modèle trop complexe. Le but n'est pas de remplacer le LLM par un autre réseau ("bazooka pour tuer un moustique"). Ce type de modèle peut donner de bons résultats en précision mais nuit à l'explicabilité et n'aide pas à comprendre le routing. Voir notamment si le projet ExpertFlow propose des idées sérieuses.

\subsection{Statistiques inter-couches}
On s'intéresse à la corrélation ou la causalité entre les paires d'experts activés dans deux couches consécutives.

Une piste est de construire une loi conjointe empirique entre deux variables aléatoires \$X\$ et \$Y\$, où \$X\$ est le couple d'experts à la couche \$L\$, et \$Y\$ à la couche \$L+1\$. L'alphabet de chaque variable est de taille 28 (choix de 2 parmi 8 experts).

\textbf{To-do} :
\begin{itemize}
\item tracer la matrice de corrélation entre \$X\$ et \$Y\$ par comptage,
\item étendre à \$Z\$ (troisième couche),
\item construire la distribution jointe empirique,
\item explorer les métriques exploitables (corrélation de Pearson, de Kendall, JSD...),
\item tester du bi-clustering pour trouver des structures de routing.
\end{itemize}

\section{Outils statistiques}

\subsection{Normalisation de la matrice de co-occurrence}
Le script \texttt{Stat\_experts.py} calcule le nombre de fois que chaque couple d'experts de la couche \$A\$ est utilisé avec un couple de la couche \$B\$. Cette matrice 28x28 peut être normalisée de plusieurs façons.

\subsubsection{Normalisation par ligne}
On obtient une probabilité conditionnelle :
\begin{center}
\textbf{P(couple\textsubscript{j} à la couche L+1 | couple\textsubscript{i} à la couche L)}
\end{center}

Cela permet de modéliser :
\begin{itemize}
\item les \textbf{transitions inter-couches},
\item le \textbf{comportement dynamique du routeur},
\item les \textbf{trajectoires de routing} dans le réseau.
\end{itemize}

\textbf{Limite} : cette normalisation peut être trompeuse. Un couple \$i\$ très rarement activé peut avoir une forte probabilité conditionnelle vers un couple \$j\$ (\textit{P(j|i) \textasciitilde{} 1}), alors que la probabilité jointe \textit{P(i \rightarrow j) = P(i) \* P(j|i)} reste très faible.

\begin{quote}
\textit{On observe une liaison forte entre \$i\$ et \$j\$, mais ce chemin est statistiquement très rare et donc peu significatif.}
\end{quote}

\subsubsection{Normalisation sur toute la matrice}
On obtient une \textbf{probabilité jointe} :
\begin{center}
\textbf{P(couple\textsubscript{i} à la couche L, couple\textsubscript{j} à la couche L+1)}
\end{center}

Chaque case \$(i, j)\$ donne la fréquence absolue du chemin \$i \rightarrow j\$.

Cette approche permet de :
\begin{itemize}
\item identifier les \textbf{transitions fréquentes},
\item \textbf{pondérer les probabilités conditionnelles} par la fréquence d'apparition de \$i\$,
\item détecter les \textbf{chemins dominants} dans le routing.
\end{itemize}

Mais elle ne permet pas d'isoler des \textbf{régularités locales} : un couple \$j\$ très fréquent apparaîtra dans plusieurs lignes, sans refléter une dépendance particulière à \$i\$.

\begin{quote}
\textit{Un couple \$j\$ peut être fréquent globalement sans dépendre du couple \$i\$ précédent.}
\end{quote}

\section{Visualisation et résultats}
On trace les matrices de co-occurrence entre couches selon les deux normalisations (conditionnelle par ligne et jointe globale), ce qui permet d'analyser les schémas de routing.

\section{Calcul de la précision de la prédiction}
On exécute le modèle sur un dataset donné (il est utile de comparer plusieurs datasets pour voir l'effet du contenu).

On enregistre, pour chaque token, les deux experts sélectionnés (top-2) dans chaque couche. Puis on vérifie si l'expert prédit est présent dans les deux experts réels sélectionnés. Cela permet de calculer la précision moyenne du système de prédiction.

\section{Résultats}
(\textit{Section à compléter avec les figures, heatmaps et tableaux de scores.})
