FROM python:3.11-slim

# scipy/scikit-learn/shap occasionally need to build from source on platforms
# without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY data/README.md ./data/README.md

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Mount the downloaded dataset at runtime, e.g.:
#   docker run -p 8888:8888 -v "$(pwd)/data/raw:/app/data/raw" <image>
VOLUME /app/data/raw

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app"]
