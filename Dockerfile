# =================================================================
# worker-vllm + ndimensional/qwen3.5-9b-uncensored-safetensors
# Con cache de compilación Inductor horneado (cold start ~25-30s)
# =================================================================

FROM runpod/worker-v1-vllm:v2.22.4

# 1. Instalar huggingface_hub
RUN pip install -q "huggingface_hub[cli]"

# 2. Descargar el modelo (solo safetensors)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='ndimensional/qwen3.5-9b-uncensored-safetensors', \
    local_dir='/models/qwen3.5-9b', \
    ignore_patterns=['*.bin', '*.pt'], \
)"

# 3. Copiar y descomprimir cache de compilación
# Debe existir vllm-compile-cache.tar.gz junto al Dockerfile
COPY vllm-compile-cache.tar.gz /tmp/

RUN mkdir -p /vllm-cache/compile \
    && tar -xzf /tmp/vllm-compile-cache.tar.gz -C /vllm-cache/compile \
    && rm -f /tmp/vllm-compile-cache.tar.gz

COPY vllm-cache/compile/ /root/.cache/vllm/


# 4. Variables de entorno
ENV MODEL_NAME="/models/qwen3.5-9b"
ENV SERVED_MODEL_NAME="ndimensional/qwen3.5-9b-uncensored-safetensors"
ENV BASE_PATH="/models"
ENV HF_HOME="/models/huggingface"
ENV MAX_MODEL_LEN="4096"
ENV GPU_MEMORY_UTILIZATION="0.9"
ENV DTYPE="bfloat16"
ENV MAX_NUM_SEQS="256"
ENV VLLM_CACHE_ROOT="/root/.cache/vllm"
