import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dvclive import Live
import joblib

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

params = yaml.safe_load(open("params.yaml"))["train"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"],
    max_depth=params["max_depth"],
    random_state=42
)

with Live(dir="live", resume=False) as live:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    live.log_metric("accuracy", acc)

    # Save model manually (still needed)
    joblib.dump(model, "model.pkl")