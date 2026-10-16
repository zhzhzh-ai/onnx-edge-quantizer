FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY onnx_edge_quantizer ./onnx_edge_quantizer
RUN pip install --no-cache-dir '.[convert]' \
    && pip install --no-cache-dir 'ai-edge-quantizer-nightly==0.10.0.dev20261016' \
    && pip install --no-cache-dir --force-reinstall --no-deps 'ai-edge-litert-nightly==2.4.0.dev20261014'
WORKDIR /work
ENTRYPOINT ["onnx-edge-quantizer"]
