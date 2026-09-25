# PaddleOCR-VL 1.6 service: official PaddleX pipeline + serving app, layout
# detection on CPU (PyTorch), VLM calls go to the Runpod vLLM endpoint.
FROM python:3.12-slim

# opencv (required by PaddleX) needs libGL/glib
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache \
      "paddlex[ocr,genai-client,serving]==3.7.2" \
      "transformers>=5,<6" \
      "torch==2.14.*" "torchvision" \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# PaddleX serializes pipeline calls to dodge a Paddle Inference bug; layout runs on
# PyTorch here, so files of a batch can overlap (verified: identical layouts, ~16% faster)
ENV PADDLE_PDX_SERVING_SERIAL_PIPELINE_CALLS=False \
    PADDLE_PDX_MODEL_SOURCE=huggingface \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    HF_HOME=/models/hf \
    PADDLE_PDX_CACHE_HOME=/models/paddlex

# bake the layout model into the image so containers start without downloads
RUN python -c "from paddlex import create_model; create_model('PP-DocLayoutV3', engine='transformers', device='cpu')"

WORKDIR /app
COPY app.py pipeline.yaml ./
EXPOSE 8080
CMD ["python", "app.py"]
