# Leakproof server. Also the on-prem image for Enterprise customers: data never leaves their network.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LEAKPROOF_DATA_DIR=/data
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY leakproof ./leakproof
COPY app ./app
RUN pip install --no-cache-dir ".[server]"
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
