# ================================================
# Titanic Survival Prediction
# Author: Rahul Gupta
# Tools: Python, Pandas, Scikit-learn, Matplotlib
# Best Model: Decision Tree — Accuracy = 77.65%
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── Step 1 — Load Dataset ──────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Shape:", df.shape)

# ── Step 2 — EDA ───────────────────────────────
print(df.info())
print(df.isnull().sum())

# ── Step 3 — Handle Missing Values ────────────
df["Age"].fillna(df["Age"].mean(), inplace=True)
df.drop(columns=["Cabin"], inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# ── Step 4 — Convert Text to Numbers ──────────
df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# ── Step 5 — Features and Target ──────────────
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[features]
y = df["Survived"]

# ── Step 6 — Train Test Split ─────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training rows:", X_train.shape)
print("Testing rows:",  X_test.shape)

# ── Step 7 — Train All 4 Models ───────────────
models = {
    "Decision Tree":  DecisionTreeClassifier(random_state=42),
    "Logistic Reg.":  LogisticRegression(max_iter=1000),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":            KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred        = model.predict(X_test)
    acc         = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name:20s} -> {acc:.2%}")

# ── Step 8 — Best Model Evaluation ────────────
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ── Step 9 — Visualize Model Comparison ───────
names  = list(results.keys())
scores = [v * 100 for v in results.values()]

plt.figure(figsize=(8, 4))
bars = plt.bar(names, scores, color=["steelblue", "orange", "green", "red"])
plt.ylim(70, 90)
plt.title("Model Comparison — Titanic Survival Prediction")
plt.ylabel("Accuracy (%)")

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{score:.1f}%",
        ha="center", fontsize=10
    )

plt.tight_layout()
plt.show()

# ── Step 10 — Predict New Passengers ──────────
# Poor young male — 3rd class:
poor_male = np.array([[3, 0, 22, 1, 0, 7.25, 0]])
print("Poor male prediction:", rf_model.predict(poor_male)[0])

# Rich older female — 1st class:
rich_lady = np.array([[1, 1, 30, 0, 0, 100, 1]])
print("Rich lady prediction:", rf_model.predict(rich_lady)[0])
