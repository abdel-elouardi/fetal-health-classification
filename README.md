# 🏥 Fetal Health Classification

Prédiction de la santé fœtale à partir de données CTG.

## 📊 Dataset
- 2126 patients
- 21 features médicales
- 3 classes : Normal, Suspect, Pathologique

## 🤖 Résultats
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
pip install -r requirements.txt
python main.py
streamlit run app.py

## 🛠️ Technologies
Python 3.11 | Scikit-learn | XGBoost | LightGBM | FastAPI | Streamlit