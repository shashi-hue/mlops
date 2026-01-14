FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    numpy \
    fastapi \
    uvicorn

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
