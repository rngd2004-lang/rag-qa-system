FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

RUN mkdir -p /app/data/uploads /app/data/chroma_db

EXPOSE 8000

CMD ["python", "main.py"]
