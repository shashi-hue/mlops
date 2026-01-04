FROM python:3.12-slim AS builder

WORKDIR /wheels

COPY requirements.txt .

RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/*

COPY train.py .

CMD ["python", "train.py"]