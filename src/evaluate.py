import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

model = joblib.load("model.pkl")  # ✅ correct loader

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

with open("metrics.txt", "w") as f:
    f.write(f"accuracy: {accuracy}\n")
