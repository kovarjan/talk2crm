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

COPY . /app

EXPOSE 8011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011", "--no-access-log"]
