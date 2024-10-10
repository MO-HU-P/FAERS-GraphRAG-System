FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data_import.py .
COPY data_embedding.py .

ARG NEO4J_URL
ARG NEO4J_USERNAME
ARG NEO4J_PASSWORD
ARG OLLAMA_URL

ENV NEO4J_URL=$NEO4J_URL
ENV NEO4J_USERNAME=$NEO4J_USERNAME
ENV NEO4J_PASSWORD=$NEO4J_PASSWORD
ENV OLLAMA_URL=$OLLAMA_URL

CMD ["sh", "-c", "python data_import.py && python data_embedding.py"]
