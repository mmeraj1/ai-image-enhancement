FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AI_IMAGE_MODEL_CACHE=/opt/models TF_CPP_MIN_LOG_LEVEL=2 \
    PRELOAD_DEBLUR=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt requirements-hf.txt ./
RUN pip install --no-cache-dir -r requirements-hf.txt requests
COPY scripts/setup_gan_models.py scripts/setup_gan_models.py
RUN python scripts/setup_gan_models.py --only deblur && chmod -R a+rX /opt/models
COPY app ./app
RUN useradd -m -u 1000 appuser
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
