FROM wlsdml1114/engui_base_128_blackwell_13:1.2 AS runtime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV COMFYUI_PATH=/opt/ComfyUI \
    HF_HOME=/opt/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Instalar ComfyUI
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH}

WORKDIR ${COMFYUI_PATH}

RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Herramientas de descarga y SDK RunPod
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir -U "huggingface_hub[cli]" hf_transfer runpod safetensors --ignore-installed

# ---------------------------------------------------------------------------
# Workflow-to-API converter
# ---------------------------------------------------------------------------
RUN cd ${COMFYUI_PATH}/custom_nodes && \
    git clone --depth 1 https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git

COPY ui_workflow_source.json ${COMFYUI_PATH}/ui_workflow_source.json

# ---------------------------------------------------------------------------
# Descarga y conversión de DavidAU/Qwen3.5-4B-Deckard-HERETIC-UNCENSORED-Thinking
# a un único archivo .safetensors (compatibilidad directa con CLIPLoader)
# ---------------------------------------------------------------------------
ENV MODEL_REPO=DavidAU/Qwen3.5-4B-Deckard-HERETIC-UNCENSORED-Thinking
ENV TEXT_ENCODERS_DIR=${COMFYUI_PATH}/models/text_encoders/qwen

RUN mkdir -p ${TEXT_ENCODERS_DIR} && \
    hf download ${MODEL_REPO} --local-dir /tmp/qwen_dl --exclude "*.bin" "*.pth" && \
    python3 -c ' \
import glob, os \
from safetensors import safe_open \
from safetensors.torch import save_file \
shards = sorted(glob.glob("/tmp/qwen_dl/*.safetensors")) \
tensors = {} \
for s in shards: \
    with safe_open(s, framework="pt", device="cpu") as f: \
        for k in f.keys(): \
            tensors[k] = f.get_tensor(k) \
save_file(tensors, "'"${TEXT_ENCODERS_DIR}"'/qwen3.5_4b_heretic.safetensors") \
' && \
    rm -rf /tmp/qwen_dl

# ---------------------------------------------------------------------------
# Handler serverless + workflow por defecto
# ---------------------------------------------------------------------------
COPY handler.py ${COMFYUI_PATH}/handler.py
COPY workflow_api.json ${COMFYUI_PATH}/workflow_api.json

WORKDIR ${COMFYUI_PATH}
CMD ["python3", "-u", "handler.py"]