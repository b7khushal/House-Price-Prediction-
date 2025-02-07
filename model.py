import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("Chennai house.csv")

# Select only the required columns
columns_to_use = ["AREA", "INT_SQFT", "DIST_MAINROAD", "N_BEDROOM",
                "N_BATHROOM", "N_ROOM", "SALE_COND", "PARK_FACIL", "SALES_PRICE"]
df = df[columns_to_use]

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical variables
categorical_cols = ["AREA", "SALE_COND", "PARK_FACIL"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])

# Standardize numerical data
scaler = StandardScaler()
numerical_cols = ["INT_SQFT", "DIST_MAINROAD", "N_BEDROOM", "N_BATHROOM", "N_ROOM"]
scaled_features = scaler.fit_transform(df[numerical_cols])

# Combine features
X = np.hstack((scaled_features, encoded_features))
y = df["SALES_PRICE"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Save model & transformers
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model training complete and saved.")