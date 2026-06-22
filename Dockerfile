# ---- Stage 1: compile CTranslate2 with Maxwell (SM 5.0) support ----
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential python3.10-dev python3-pip python3-distutils \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG CT2_VERSION=v4.4.0
ARG CUDA_ARCH_LIST="5.0"

# Download CTranslate2 source + submodules (requires --network=host at build time)
RUN curl -fsSL "https://github.com/OpenNMT/CTranslate2/archive/refs/tags/${CT2_VERSION}.tar.gz" \
        | tar xz -C / \
    && mv /CTranslate2-${CT2_VERSION#v} /ct2 \
    && cd /ct2 \
    && mkdir -p third_party/spdlog third_party/cpu_features third_party/ruy \
    && curl -fsSL "https://github.com/gabime/spdlog/archive/refs/heads/v1.x.tar.gz" \
        | tar xz --strip-components=1 -C third_party/spdlog \
    && curl -fsSL "https://github.com/google/cpu_features/archive/refs/heads/main.tar.gz" \
        | tar xz --strip-components=1 -C third_party/cpu_features \
    && curl -fsSL "https://github.com/google/ruy/archive/refs/heads/master.tar.gz" \
        | tar xz --strip-components=1 -C third_party/ruy \
    && mkdir -p third_party/ruy/third_party/cpuinfo \
    && curl -fsSL "https://github.com/pytorch/cpuinfo/archive/refs/heads/main.tar.gz" \
        | tar xz --strip-components=1 -C third_party/ruy/third_party/cpuinfo

COPY patch-awq-sm52.py /tmp/patch-awq-sm52.py
RUN python3 /tmp/patch-awq-sm52.py \
        /ct2/src/ops/awq/dequantize_gpu.cu \
        /ct2/src/ops/awq/gemv_gpu.cu \
        /ct2/src/ops/awq/gemm_gpu.cu

WORKDIR /ct2

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON \
        -DWITH_MKL=OFF -DWITH_RUY=ON -DOPENMP_RUNTIME=COMP \
        -DCUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
        -DBUILD_CLI=OFF \
    && cmake --build build --target install -j"$(nproc)" \
    && ldconfig

# Build the Python wheel
WORKDIR /ct2/python
RUN pip3 install --no-cache-dir wheel setuptools pybind11 \
    && python3 setup.py bdist_wheel

# ---- Stage 2: runtime ----
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Graft the custom CTranslate2 build over the pip-installed version
COPY --from=builder /usr/local/lib/libctranslate2.so* /usr/local/lib/
COPY --from=builder /ct2/python/dist/*.whl /tmp/
RUN pip install --force-reinstall --no-deps /tmp/*.whl \
    && rm -f /tmp/*.whl \
    && ldconfig

COPY app/ app/
COPY static/ static/

RUN useradd -m -s /bin/bash appuser
RUN mkdir -p /tmp/whisper-stt && chown appuser:appuser /tmp/whisper-stt
RUN mkdir -p /cache && chown appuser:appuser /cache
ENV HF_HOME=/cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
