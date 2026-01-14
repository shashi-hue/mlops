import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

mlflow.set_experiment("fraud-model")

with mlflow.start_run():
    mlflow.log_metric("auc", auc)
    mlflow.sklearn.log_model(model, "model")

    # Quality gate
    if auc < 0.88:
        raise ValueError("AUC below threshold")

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "fraud_detection_model"
    )