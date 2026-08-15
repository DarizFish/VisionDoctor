FROM python:3.11-slim-bookworm

ARG NUMPY_VERSION=2.4.6

RUN python -m pip install --no-cache-dir "numpy==${NUMPY_VERSION}" \
    && useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin sandbox

ENV HOME=/tmp \
    PYTHONIOENCODING=utf-8 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /workspace
USER 65532:65532
