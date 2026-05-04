import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CHARGER
df = pd.read_csv('data/raw/fetal_health.csv')
print(f"✅ {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 2. NETTOYAGE
df.columns = df.columns.str.strip()
for col in df.columns:
    if df[col].dtype == object:
        try: df[col] = pd.to_numeric(df[col])
        except: pass
df = df.drop_duplicates()
df = df.drop(columns=df.columns[df.isnull().mean() > 0.5])
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median() if abs(df[col].skew()) > 1 else df[col].mean())
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'fetal_health': continue
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    df = df[df[col].between(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))]
for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'fetal_health': continue
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col] - df[col].min())
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
corr = df.drop(columns=['fetal_health']).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
df = df.drop(columns=[c for c in upper.columns if any(upper[c] > 0.9)])
print(f"✅ Nettoyage : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Corrélation")
plt.tight_layout()
plt.savefig("models/correlation.png")

# 3. SPLIT
X = df.drop(columns=['fetal_health'])
y = df['fetal_health'] - 1  # ← XGBoost veut 0,1,2 pas 1,2,3
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. NORMALISER
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 5. MODÈLES
models = {
    "Random Forest":       RandomForestClassifier(random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "XGBoost":             XGBClassifier(random_state=42, eval_metric='mlogloss'),
    "LightGBM":            LGBMClassifier(random_state=42, verbose=-1),
}

resultats = {}
for nom, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    resultats[nom] = score
    print(f"  {nom:25s} : {score:.4f}")

# 6. MEILLEUR MODÈLE
meilleur = max(resultats, key=resultats.get)
print(f"\n🏆 Meilleur : {meilleur} ({resultats[meilleur]:.4f})")
print(classification_report(y_test, models[meilleur].predict(X_test)))

# 7. GRAPHIQUES
plt.figure(figsize=(10, 5))
plt.barh(list(resultats.keys()), list(resultats.values()), color="steelblue")
plt.title("Comparaison des modèles")
plt.tight_layout()
plt.savefig("models/comparaison.png")

cm = confusion_matrix(y_test, models[meilleur].predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.title(f"Confusion — {meilleur}")
plt.tight_layout()
plt.savefig("models/confusion.png")
print("✅ Graphiques sauvegardés")
