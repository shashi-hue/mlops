
from prefect import task, flow
import mlflow
import hashlib
import os
import json
os.environ["PREFECT__SERVER__ENABLED"] = "false"
@task
def load_data():
    data = {"samples": 1000, "features": 20}
    return data

@task
def data_hash(data):
    return hashlib.md5(json.dumps(data).encode()).hexdigest()

@task(retries=3, retry_delay_seconds=30#, cache_key_fn= lambda ctx, params: params["hash_id"]
      )
def train(hash_id):
    model_path = f"models/{hash_id}/model.pt"

    if os.path.exists(model_path):
        return model_path
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        f.write(b"dummy model")

    return model_path

@task
def register(model_path, hash_id):
    mlflow.start_run()
    mlflow.log_artifact(model_path)
    mlflow.set_tag("data_hash", hash_id)
    mlflow.end_run()

@flow
def pipeline():
    data = load_data()
    hash_id = data_hash(data)
    model_path = train(hash_id=hash_id)
    register(model_path, hash_id)

if __name__ == "__main__":
    pipeline()


