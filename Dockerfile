
FROM --platform=linux/amd64 python:3.9-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt  ./
COPY Download.py       ./
COPY load_model.py     ./
COPY app.py            ./
COPY models.py         ./
COPY face_landmarker.task ./

COPY model   /app/model/
COPY mlflow  /app/mlflow/

COPY static     /app/static/
COPY templates  /app/templates/
COPY utils      /app/utils/


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python load_model.py

ENV PORT=8080
ENV INGEST_TOKEN=changeme  

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]
    