#---------6---------
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Preprocess: select relevant columns and drop rows with missing values
data = df[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()
data['sex'] = data['sex'].map({'male': 0, 'female': 1})  # Encode categorical

# Features and target
X = data[['pclass', 'sex', 'age', 'fare']]
y = data['survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Naive Bayes Accuracy on Titanic dataset: {acc:.2f}")

# Predict on new data example
sample = pd.DataFrame([[2, 1, 28.0, 80.0]], columns=['pclass', 'sex', 'age', 'fare'])
print("Sample Prediction (class 2, female, age 28, fare 80):", model.predict(sample)[0])