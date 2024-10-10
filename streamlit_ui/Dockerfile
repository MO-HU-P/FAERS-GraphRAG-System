FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ARG NEO4J_URL
ARG NEO4J_USERNAME
ARG NEO4J_PASSWORD
ARG OPENAI_API_KEY
ARG OLLAMA_URL

ENV NEO4J_URL=$NEO4J_URL
ENV NEO4J_USERNAME=$NEO4J_USERNAME
ENV NEO4J_PASSWORD=$NEO4J_PASSWORD
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV OLLAMA_URL=$OLLAMA_URL

EXPOSE 8501

CMD ["sh", "-c", "while ! nc -z neo4j 7687; do sleep 1; done; streamlit run main.py"]

