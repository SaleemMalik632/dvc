import pandas as pd
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from dvclive import Live

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()  # Ensure y_test is a Series
model = joblib.load("model.pkl")

# Make predictions
preds = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="macro")
recall = recall_score(y_test, preds, average="macro")
f1 = f1_score(y_test, preds, average="macro")

# Save metrics
os.makedirs("eval", exist_ok=True)
with open("eval/metrics.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }, f, indent=4)

# Log confusion matrix using dvclive (modern log_plot format)
confusion_data = pd.DataFrame({
    "Actual": y_test.astype(str),
    "Predicted": preds.astype(str)
})

with Live() as live:
    live.log_metric("accuracy", accuracy)
    live.log_metric("precision", precision)
    live.log_metric("recall", recall)
    live.log_metric("f1", f1)

    # Log confusion matrix
    live.log_plot(
        name="confusion_matrix",
        datapoints=confusion_data,
        x="Actual",
        y="Predicted",
        template="confusion",
        title="Confusion Matrix",
        x_label="Actual Label",
        y_label="Predicted Label"
    )

    # # Log ROC and PRC curves
    # classes = sorted(y_test.unique())
    # y_test_bin = label_binarize(y_test, classes=classes)
    # preds_bin = label_binarize(preds, classes=classes)

    # fpr, tpr, _ = roc_curve(y_test_bin.ravel(), preds_bin.ravel())
    # prc_prec, prc_rec, _ = precision_recall_curve(y_test_bin.ravel(), preds_bin.ravel())

    # roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    # prc_data = pd.DataFrame({"precision": prc_prec, "recall": prc_rec})

    # live.log_plot("roc_curve", datapoints=roc_data, x="fpr", y="tpr", title="ROC Curve")
    # live.log_plot("precision_recall_curve", datapoints=prc_data, x="recall", y="precision", title="Precision-Recall Curve")
