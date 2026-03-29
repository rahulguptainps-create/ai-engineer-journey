# ml first project
# ================================================
# California Housing Price Prediction
# Author: Rahul Gupta
# Tools: Python, Pandas, Scikit-learn, Matplotlib
# Best Model: Random Forest — R2=0.81, MAE=0.33
# ================================================

# NOTE: If you get Error 403, run this first:
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# California Housing Price Prediction
# Best Model: Random Forest — R2=0.81 | MAE=0.33

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset — built in
housing = fetch_california_housing()

# Convert to DataFrame:
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target   # target column add karo
                               

print(df.shape)
print(df.head())
print(df.columns.tolist())
df.info()
df.describe()
df.isnull().sum()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#STEP 2 PLOT THE GRAPH

# Graph 1 — Distribution of house prices:
# Hint: axes[0].hist(df["Price"], bins=50, color="steelblue")
axes[0].hist(df["Price"],bins=50, color = "steelblue" )
axes[0].set_title("House Price Distribution")
axes[0].set_xlabel("price")
axes[0].set_ylabel("Frequency")
# Graph 2 — Income vs Price scatter plot:

axes[1].scatter(df["MedInc"],df["Price"],alpha=0.3,color="green")
axes[1].set_title("Income vs Price")
axes[1].set_xlabel("MedInc")
axes[1].set_ylabel("Price")
# Graph 3 — House Age distribution:

axes[2].hist(df["HouseAge"], bins=30, color="orange")
axes[2].set_title("House Age Distribution")
axes[2].set_xlabel("HouseAge")
axes[2].set_ylabel("Frequency")

# Add titles and labels yourself — you know how!

plt.tight_layout()
plt.show()

# STEP 3 FEACTURES SELECTION AND TERANING

# Step 1 — Select features (X) and target (y):
# All columns except Price are features!
features = ["MedInc", "HouseAge", "AveRooms",
            "AveBedrms", "Population", "AveOccup",
            "Latitude", "Longitude"]

X = df[features]        # fill in — features list use karo
y = df["Price"]      # fill in — target column kaunsa hai?

# Step 2 — Train test split:
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)
print("Training rows:", X_train.shape)
print("Testing rows:",  X_test.shape)

# Step 3 — Train TWO models:

# Model 1 — Linear Regression:
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Model 2 — Random Forest Regressor:
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Step 4 — Evaluate both models:
print("Linear Regression:")
print(f"  MAE:  {mean_absolute_error(y_test, lr_pred):.2f}")
print(f"  R²:   {r2_score(y_test, lr_pred):.2f}")

print("Random Forest:")
print(f"  MAE:  {mean_absolute_error(y_test, rf_pred):.2f}")
print(f"  R²:   {r2_score(y_test, rf_pred):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_test, lr_pred, alpha=0.3, color="steelblue")
axes[0].plot([0, 5], [0, 5], color="red", linewidth=2)
axes[0].set_title("Linear Regression — Actual vs Predicted")
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")

axes[1].scatter(y_test, rf_pred, alpha=0.3, color="green")
axes[1].plot([0, 5], [0, 5], color="red", linewidth=2)
axes[1].set_title("Random Forest — Actual vs Predicted")
axes[1].set_xlabel("Actual Price")
axes[1].set_ylabel("Predicted Price")

plt.tight_layout()
plt.show()
