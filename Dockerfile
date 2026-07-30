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
    hf download ${MODEL_REPO} --local-dir /tmp/qwen_dl --include "*.safetensors" && \
    python3 -c 'import glob, os; from safetensors import safe_open; from safetensors.torch import save_file; shards = sorted(glob.glob("/tmp/qwen_dl/*.safetensors")); tensors = {k: f.get_tensor(k) for s in shards for f in [safe_open(s, framework="pt", device="cpu")] for k in f.keys()}; save_file(tensors, os.path.join(os.environ["TEXT_ENCODERS_DIR"], "qwen3.5_4b_heretic.safetensors"))' && \
    rm -rf /tmp/qwen_dl

# ---------------------------------------------------------------------------
# Fish Audio S2-Pro (TTS + voice cloning)
# ---------------------------------------------------------------------------
RUN cd ${COMFYUI_PATH}/custom_nodes && \
    git clone --depth 1 https://github.com/saganaki22/ComfyUI-FishAudioS2.git && \
    cd ComfyUI-FishAudioS2 && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps descript-audio-codec "descript-audiotools>=0.7.2" && \
    pip install --no-cache-dir flatten-dict importlib-resources julius randomname ffmpy argbind

ENV FISH_MODEL_REPO=fishaudio/s2-pro
ENV FISH_MODEL_DIR=${COMFYUI_PATH}/models/fishaudioS2/s2-pro

RUN mkdir -p ${FISH_MODEL_DIR} && \
    hf download ${FISH_MODEL_REPO} --local-dir ${FISH_MODEL_DIR}

RUN test -d ${FISH_MODEL_DIR} && \
    size_mb=$(du -sm ${FISH_MODEL_DIR} | cut -f1) && \
    echo "Fish Audio S2-Pro descargado: ${size_mb}MB" && \
    [ "$size_mb" -gt 5000 ] || (echo "ERROR: el modelo de Fish Audio no se descargó completo" && exit 1)

# ---------------------------------------------------------------------------
# Handler serverless + workflow por defecto
# ---------------------------------------------------------------------------
COPY handler.py ${COMFYUI_PATH}/handler.py
COPY workflow_api.json ${COMFYUI_PATH}/workflow_api.json

WORKDIR ${COMFYUI_PATH}
CMD ["python3", "-u", "handler.py"]