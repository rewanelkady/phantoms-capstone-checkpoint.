import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# Step 1:

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df_raw = pd.read_csv(url, sep=';')

conn = sqlite3.connect('dataset.db')
df_raw.to_sql('wine_data', conn, if_exists='replace', index=False)
print("Done")

df = pd.read_sql_query("SELECT * FROM wine_data", conn)
conn.close()

# Step 2:

df = df.dropna()
df['target'] = np.where(df['quality'] >= 6, 1, 0)

X = df.drop(columns=['quality', 'target'])
y = df['target']

print(f"عدد الـ Features: {X.shape[1]}")
print(f"توزيع الـ Label (Target):\n{y.value_counts()}")

# Step 3:

print("\n--- المرحلة 3: تدريب النماذج ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# نموذج 1: Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# نموذج 2: Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_clf.fit(X_train, y_train)

# Step 4:

def evaluate_model(name, model, X_t, y_t):
    preds = model.predict(X_t)
    print(f"\n================ {name} ================")
    print("Confusion Matrix:")
    print(confusion_matrix(y_t, preds))
    print("\nClassification Report:")
    print(classification_report(y_t, preds))
    return f1_score(y_t, preds)

f1_log = evaluate_model("Logistic Regression", log_reg, X_test_scaled, y_test)
f1_dt = evaluate_model("Decision Tree Classifier", dt_clf, X_test, y_test)

print("\n--- القرار النهائي ---")
if f1_dt > f1_log:
    print(f"النموذج الأفضل هو Decision Tree بمعدل F1-Score = {f1_dt:.2f}")
else:
    print(f"النموذج الأفضل هو Logistic Regression بمعدل F1-Score = {f1_log:.2f}")
