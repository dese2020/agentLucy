FROM wlsdml1114/engui_base_128_blackwell_13:1.2 AS runtime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV COMFYUI_PATH=/workspace/ComfyUI \
    HF_HOME=/workspace/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR ${COMFYUI_PATH}

# ---------------------------------------------------------------------------
# Herramientas para descargar de HuggingFace (hf CLI moderno + acelerador)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir -U "huggingface_hub[cli]" hf_transfer runpod --ignore-installed

# ---------------------------------------------------------------------------
# NO hace falta custom node: el workflow usa los nodos nativos de ComfyUI
# CLIPLoader + TextGenerate (comfy-core >= 0.19), que ya vienen en la imagen
# base.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Descarga del modelo. Usamos la versión oficial de Comfy-Org, que ya viene
# empaquetada en el formato single-file que CLIPLoader espera (el mismo
# archivo que aparece hardcodeado en tu workflow: qwen3.5_4b_bf16.safetensors).
# Así nos ahorramos merges/conversiones de formato.
# ---------------------------------------------------------------------------
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV MODEL_REPO=Comfy-Org/Qwen3.5
ENV MODEL_FILE=text_encoders/qwen3.5_4b_bf16.safetensors
ENV TEXT_ENCODERS_DIR=${COMFYUI_PATH}/models/text_encoders/qwen

RUN mkdir -p ${TEXT_ENCODERS_DIR} && \
    hf download ${MODEL_REPO} ${MODEL_FILE} \
        --local-dir /tmp/qwen_dl \
        --local-dir-use-symlinks False && \
    mv /tmp/qwen_dl/${MODEL_FILE} ${TEXT_ENCODERS_DIR}/qwen3.5_4b_bf16.safetensors && \
    rm -rf /tmp/qwen_dl

# ---------------------------------------------------------------------------
# Handler serverless + workflow por defecto
# ---------------------------------------------------------------------------
COPY handler.py ${COMFYUI_PATH}/handler.py
COPY workflow_api.json ${COMFYUI_PATH}/workflow_api.json

WORKDIR ${COMFYUI_PATH}
CMD ["python3", "-u", "handler.py"]
