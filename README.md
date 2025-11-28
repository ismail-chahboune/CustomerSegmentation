# Customer Segmentation using KMeans and PCA

##  Description du Projet
Ce projet effectue une **segmentation client** à partir d’un dataset marketing.  
Il utilise des techniques de **clustering non supervisé** (KMeans) et de **réduction de dimension** (PCA) pour analyser les comportements des clients et identifier des groupes similaires.

---

##  Objectifs du Projet
- Nettoyer et préparer les données clients  
- Créer de nouvelles variables pertinentes (âge, enfants, dépenses totales, campagnes acceptées…)  
- Visualiser la distribution et corrélation des variables  
- Effectuer un clustering avec KMeans pour identifier des segments clients  
- Réduire la dimension des données avec PCA pour visualiser les clusters  
- Sauvegarder le modèle KMeans et le scaler pour usage futur (`kmeans_model.pkl`, `scaler.pkl`)  

---

##  Approche
1. Chargement et exploration des données (`customer_segmentation.csv`)  
2. Prétraitement : gestion des valeurs manquantes et création de nouvelles colonnes  
3. Analyse exploratoire : histogrammes, boxplots, heatmaps, barplots  
4. Clustering KMeans :
   - Standardisation des données  
   - Méthode du coude pour déterminer le nombre optimal de clusters  
   - Attribution des clusters aux clients  
5. Réduction de dimension avec PCA pour visualisation 2D  
6. Sauvegarde des objets (`scaler` et `kmeans_model`) pour prédiction future  

---

##  Résultats
- Segmentation en 6 clusters identifiés  
- Visualisation PCA des clusters  
- Résumé statistique par cluster  
- Graphiques exploratoires pour comprendre les caractéristiques des clients  

---

##  Fichiers Principaux
- `customer_segmentation.py` — Script complet de traitement, clustering et visualisation  
- `scaler.pkl` — Scaler pour normalisation des données  
- `kmeans_model.pkl` — Modèle KMeans sauvegardé  
- Dataset `customer_segmentation.csv`  
- `README.md` — Documentation  

---

##  Auteur
Projet réalisé par **Chahboune Ismail**
