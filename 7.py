#-----------------7-----------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# --- 1. Load Glass dataset from CSV file ---
data = pd.read_csv("./glass.csv")  # Change this to your file path if needed

# --- 2. Prepare features and labels ---
X = data.drop(columns=['Type'])  # Features
y = data['Type']                 # Target

# Normalize data for fair distance comparison
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Split data (70% train, 30% test) ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- 4. Define distance functions ---
def euclidean(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def manhattan(p1, p2):
    return np.sum(np.abs(p1 - p2))

# --- 5. KNN Implementation ---
def knn(X_train, y_train, x_test, k=3, dist='euclidean'):
    distances = []
    for i in range(len(X_train)):
        d = euclidean(x_test, X_train[i]) if dist == 'euclidean' else manhattan(x_test, X_train[i])
        distances.append((d, y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = [label for _, label in distances[:k]]
    return max(set(neighbors), key=neighbors.count)

# --- 6. Run KNN for all test samples ---
def evaluate_knn(X_train, y_train, X_test, y_test, dist):
    preds = [knn(X_train, y_train, x, k=3, dist=dist) for x in X_test]
    acc = accuracy_score(y_test, preds)
    print(f"KNN Accuracy ({dist.title()} Distance): {acc:.2f}")

# --- 7. Evaluate both Euclidean and Manhattan ---
evaluate_knn(X_train, y_train, X_test, y_test, dist='euclidean')
evaluate_knn(X_train, y_train, X_test, y_test, dist='manhattan')