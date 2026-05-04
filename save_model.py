import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Charger et nettoyer les données
df = pd.read_csv('data/raw/fetal_health.csv')
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.drop(columns=df.columns[df.isnull().mean() > 0.5])
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median() if abs(df[col].skew()) > 1 else df[col].mean())
for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'fetal_health': continue
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    df = df[df[col].between(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))]
for col in df.select_dtypes(include=[np.number]).columns:
    if col == 'fetal_health': continue
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col] - df[col].min())
corr = df.drop(columns=['fetal_health']).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
df = df.drop(columns=[c for c in upper.columns if any(upper[c] > 0.9)])

# Split et normaliser
X = df.drop(columns=['fetal_health'])
y = (df['fetal_health'] - 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Entraîner et sauvegarder
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Modèle sauvegardé → models/model.pkl")
print("✅ Scaler sauvegardé → models/scaler.pkl")
