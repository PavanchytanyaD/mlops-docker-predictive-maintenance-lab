# Stage 1: Build and Train the Model
FROM python:3.11-slim AS model_training

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ /app/data/
COPY src/features.py src/model_training.py /app/
RUN python model_training.py


# Stage 2: Serve Predictions
FROM python:3.11-slim AS serving

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Only the trained model and its metadata come from stage 1, not the data
COPY --from=model_training /app/model /app/model
COPY src/features.py src/main.py /app/
COPY src/templates /app/templates
COPY src/statics /app/statics

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 80
# Docker "exposes" port 80 because it is the standard, well-known port for the HTTP (web) protocol

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:80/health')"

# gunicorn instead of the Flask development server
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "2", "main:app"]
