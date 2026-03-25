# 🧪 A/B Testing — Landing Page Optimization

Expérimentation contrôlée pour comparer deux versions d'une landing page.  
**Projet 2 — Data Science**

---

## 🚀 Démo en ligne

> Déployez sur [share.streamlit.io](https://share.streamlit.io) (cf. section déploiement)

---

## 📂 Structure du projet

```
ab_testing/
│
├── 📓 EDA_AB_Testing.ipynb          # Notebook — analyse complète pas à pas
├── 🐍 app.py                         # Application Streamlit interactive
├── 🐍 generate_notebook.py           # Script pour générer le notebook
├── 📊 ab_data.csv                    # Dataset (auto-généré ou Kaggle)
├── 📋 requirements.txt
├── 📖 README.md
├── plots/
│   ├── 01_ab_overview.png
│   ├── 02_segment_analysis.png
│   └── 03_temporal_evolution.png
└── .streamlit/
    └── config.toml                   # Thème sombre personnalisé
```

---

## 📓 Notebook — EDA_AB_Testing.ipynb

| Section | Contenu |
|---|---|
| 1. Chargement & Exploration | Audit qualité, SRM check, distributions |
| 2. Design expérimental | Vérification équilibre des groupes |
| 3. Tests statistiques | Chi², Z-test, t-test, Mann-Whitney U |
| 4. Visualisations | Distributions, bootstrap CI, évolution temporelle |
| 5. Analyse segmentée | Effets par device, pays, tranche d'âge |
| 6. Rapport de décision | Synthèse, recommandation, export |

```bash
# Générer puis lancer le notebook
python generate_notebook.py
jupyter notebook EDA_AB_Testing.ipynb
```

---

## 🖥️ Application Streamlit

Dashboard interactif avec filtres dynamiques (device, pays, âge) et 5 onglets :

| Onglet | Contenu |
|---|---|
| 📊 Vue d'ensemble | Métriques clés, taux de conversion, tableau récapitulatif |
| 🔬 Tests statistiques | Chi², Z-test, t-test, table de contingence |
| 📈 Visualisations | Distributions, bootstrap, évolution temporelle |
| 🔍 Analyse segmentée | Lift et p-value par sous-groupe |
| 💡 Décision | Impact business, recommandation, exports |

---

## 📦 Dataset

### Option A — Dataset Kaggle (recommandé)
Source : [Kaggle — A/B Testing](https://www.kaggle.com/datasets/zhangluyuan/ab-testing)

Renommez le fichier en `ab_data.csv` et placez-le à la racine du projet.

| Variable | Description |
|---|---|
| `user_id` | Identifiant unique utilisateur |
| `timestamp` | Date et heure de la visite |
| `group` | `control` ou `treatment` |
| `landing_page` | `old_page` ou `new_page` |
| `converted` | 0 ou 1 (variable cible) |

### Option B — Dataset synthétique (auto-généré)
Si `ab_data.csv` est absent, un dataset réaliste est généré automatiquement avec :
- 10 000 utilisateurs équirépartis
- Taux contrôle : ~11.2% | Taux traitement : ~12.7%
- Variables enrichies : device, pays, âge, clics, temps, revenu

---

## 🛠 Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/ab-testing.git
cd ab-testing

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur : `http://localhost:8501`

---

## ☁️ Déploiement sur Streamlit Cloud

```bash
# Étape 1 — Pousser sur GitHub
git init
git add .
git commit -m "feat: A/B Testing Dashboard"
git branch -M main
git remote add origin https://github.com/votre-username/ab-testing.git
git push -u origin main
```

**Étape 2** — Sur [share.streamlit.io](https://share.streamlit.io) :
1. Se connecter avec GitHub
2. "New app" → sélectionner le dépôt
3. Fichier principal : `app.py`
4. Deploy ✅

> ⚠️ Inclure `ab_data.csv` dans le dépôt GitHub pour Streamlit Cloud.

---

## 🧰 Technologies utilisées

| Outil | Usage |
|---|---|
| Python 3.11 | Langage principal |
| Pandas | Manipulation des données |
| NumPy | Calculs statistiques et bootstrap |
| SciPy | Tests statistiques (chi², t-test, z-test, Mann-Whitney) |
| Matplotlib / Seaborn | Visualisations |
| Streamlit | Interface web interactive |

---

## 🔬 Méthodologie statistique

### Hypothèses
- **H₀** : Pas de différence entre les taux de conversion (p_A = p_B)
- **H₁** : Les taux de conversion sont différents (p_A ≠ p_B)
- **Seuil α** : 0.05 (paramétrable dans le dashboard)

### Tests utilisés
1. **Chi² d'indépendance** — Test principal sur proportions
2. **Z-test** — Confirmation avec intervalle de confiance
3. **t-test de Student** — Métriques continues (temps, clics)
4. **Mann-Whitney U** — Variante non-paramétrique robuste
5. **Bootstrap** — Estimation non-paramétrique de l'incertitude
6. **SRM Check** — Vérification de l'équilibre des groupes

---

## 📊 Variables créées (Feature Engineering)

| Variable | Description |
|---|---|
| `date` | Jour de la visite |
| `hour` | Heure de la visite |
| `week` | Numéro de semaine |

---

## 🤝 Liens avec le Projet 1 (EDA Marketing)

Ce projet s'inscrit dans la continuité du Projet 1 :
- **Projet 1** : EDA exploratoire → identification des segments à fort potentiel
- **Projet 2** : A/B testing → validation expérimentale d'une hypothèse marketing
- **Projet 3 (à venir)** : Modélisation prédictive → score de propension à la conversion

---

## 📄 Licence

MIT — Libre d'utilisation pour tout projet éducatif ou professionnel.
