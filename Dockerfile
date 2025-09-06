
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects PORT; default to 10000 for local container run
ENV PORT=10000
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
