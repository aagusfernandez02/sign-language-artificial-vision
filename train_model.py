import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Leer dataset
df = pd.read_csv("dataset.csv")

# Separar features y etiquetas
X = df.iloc[:, :-1]  # columnas de keypoints (63)
y = df.iloc[:, -1]   # última columna: letra

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Crear y entrenar el modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar
acc = clf.score(X_test, y_test)
print(f"Precisión en test: {acc*100:.2f}%")

# Guardar modelo
joblib.dump(clf, "sign_language_model.pkl")
print("✅ Modelo guardado como sign_language_model.pkl")