# 🏥 Fetal Health Classification

Prédiction de la santé fœtale à partir de données CTG (Cardiotocographie).

## 📊 Dataset
- 2126 patients
- 21 features médicales
- 3 classes : Normal, Suspect, Pathologique

## 🤖 Modèles testés
| Modèle | Accuracy |
|--------|----------|
| Gradient Boosting | 97.24% 🏆 |
| XGBoost | 96.13% |
| Random Forest | 95.58% |
| Decision Tree | 92.27% |
| SVM | 91.16% |
| Logistic Regression | 90.06% |
| KNN | 90.06% |

## 🚀 Lancement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle
python main.py

# Lancer le dashboard
streamlit run app.py

# Lancer l'API
uvicorn api:app --reload
```

## 🛠️ Technologies
- Python 3.11
- Scikit-learn
- XGBoost / LightGBM
- FastAPI
- Streamlit
