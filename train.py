import numpy as np
import random
import torch
import mlflow
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-1")

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=SEED
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=SEED)

model = torch.nn.Sequential(
    torch.nn.Linear(20,1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss()

optim = torch.optim.Adam(model.parameters(), lr=0.02)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)



with mlflow.start_run():
    mlflow.log_param("seed",SEED)
    mlflow.log_param("lr",0.02)

    for epoch in range(50):
        optim.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optim.step()

    mlflow.log_metric("final_loss",loss.item())
    mlflow.pytorch.log_model(model,"model")
