# =================================================================
# worker-vllm + ndimensional/qwen3.5-9b-uncensored-safetensors
# Repo base: https://github.com/runpod-workers/worker-vllm
#
# El entrypoint y handler vienen heredados de la imagen base.
# NO necesitas definirlos aquí.
# =================================================================

FROM runpod/worker-v1-vllm:latest

# 1. Instalar huggingface_hub ANTES de descargar
RUN pip install -q "huggingface_hub[cli]"

# 2. Descargar el modelo (solo safetensors, ~18 GB)
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='ndimensional/qwen3.5-9b-uncensored-safetensors', \
    local_dir='/models/qwen3.5-9b', \
    ignore_patterns=['*.bin', '*.pt'], \
)"

# 3. Apuntar el worker al modelo local bakeado
ENV MODEL_NAME="/models/qwen3.5-9b"
ENV BASE_PATH="/models"
ENV HF_HOME="/models/huggingface"
ENV MAX_MODEL_LEN="8192"
ENV GPU_MEMORY_UTILIZATION="0.92"
ENV DTYPE="bfloat16"
ENV MAX_NUM_SEQS="256"
