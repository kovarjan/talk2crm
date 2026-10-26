FROM python:3.11-slim

ARG INSTALL_GPU_DEPS=false
ARG PIP_TIMEOUT=120
ARG PIP_RETRIES=10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       curl \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.gpu.txt /app/
RUN pip install --upgrade pip \
    && pip install --timeout ${PIP_TIMEOUT} --retries ${PIP_RETRIES} -r /app/requirements.txt \
    && if [ "${INSTALL_GPU_DEPS}" = "true" ]; then \
         pip install --timeout ${PIP_TIMEOUT} --retries ${PIP_RETRIES} -r /app/requirements.gpu.txt; \
       fi

# Run as an unprivileged user; it owns the writable runtime dirs.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/cache /app/logs /home/app/.cache/huggingface \
    && chown -R app:app /app/cache /app/logs /home/app

COPY --chown=app:app . /app

ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

USER app

EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011", "--no-access-log"]
