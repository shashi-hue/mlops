from flask import Flask, request, jsonify
import torch
import mlflow.pytorch

app = Flask(__name__)

model = None
model_loaded = False

def load_model():
    global model, model_loaded
    model = mlflow.pytorch.load_model("models:/mlops-1/Production")
    model.eval()
    model_loaded = True

@app.before_request
def startup():
    global model_loaded
    if not model_loaded:
        load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["inputs"]
    x = torch.tensor(data, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        y = model(x).tolist()

    return jsonify(y)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)