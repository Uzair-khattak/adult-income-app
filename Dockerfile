FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_DIR=/app/models API_URL=http://127.0.0.1:8000
EXPOSE 7860
CMD ["bash", "start.sh"]
