FROM wlsdml1114/engui_base_128_blackwell_13:1.2 AS runtime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# OJO: instalamos ComfyUI en /opt, NO en /workspace. Si esta imagen usa
# /workspace como punto de montaje de un Network Volume en RunPod, cualquier
# cosa que pongamos ahí durante el build se pierde en runtime (esto fue lo
# que causó el error "can't open file /workspace/ComfyUI/main.py").
ENV COMFYUI_PATH=/opt/ComfyUI \
    HF_HOME=/opt/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Instalar ComfyUI (no viene en la imagen base)
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH}

WORKDIR ${COMFYUI_PATH}

# La imagen base ya trae torch compilado para Blackwell/CUDA 13. No lo
# reinstalamos: filtramos la línea de torch del requirements.txt de ComfyUI
# antes de instalar el resto de dependencias, para no romper esa build.
#RUN grep -v -i '^torch' requirements.txt > requirements.filtered.txt && \
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Herramientas para descargar de HuggingFace (hf CLI moderno + acelerador)
# y el SDK de RunPod para el handler serverless.
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir -U "huggingface_hub[cli]" hf_transfer runpod --ignore-installed

# ---------------------------------------------------------------------------
# NO hace falta custom node: el workflow usa los nodos nativos de ComfyUI
# CLIPLoader + TextGenerate (comfy-core >= 0.19). Verificar que la versión
# clonada de ComfyUI ya incluya estos nodos (deberían estar en cualquier
# versión reciente; si el handler falla con "node type not found", puede
# que haga falta anclar a un tag/commit más nuevo con --branch).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Descarga del modelo. Usamos la versión oficial de Comfy-Org, que ya viene
# empaquetada en el formato single-file que CLIPLoader espera (el mismo
# archivo que tu workflow referencia: qwen3.5_4b_bf16.safetensors).
# ---------------------------------------------------------------------------

ENV MODEL_REPO=Comfy-Org/Qwen3.5
ENV MODEL_FILE=text_encoders/qwen3.5_4b_bf16.safetensors
ENV TEXT_ENCODERS_DIR=${COMFYUI_PATH}/models/text_encoders/qwen

RUN mkdir -p ${TEXT_ENCODERS_DIR} && \
    hf download ${MODEL_REPO} ${MODEL_FILE} \
        --local-dir /tmp/qwen_dl && \
    mv /tmp/qwen_dl/${MODEL_FILE} \
       ${TEXT_ENCODERS_DIR}/qwen3.5_4b_bf16.safetensors && \
    rm -rf /tmp/qwen_dl

# ---------------------------------------------------------------------------
# Handler serverless + workflow por defecto
# ---------------------------------------------------------------------------
COPY handler.py ${COMFYUI_PATH}/handler.py
COPY workflow_api.json ${COMFYUI_PATH}/workflow_api.json

# ---------------------------------------------------------------------------
# Fish Audio S2-Pro (TTS + voice cloning), custom node de la comunidad.
# https://github.com/Saganaki22/ComfyUI-FishAudioS2
# ---------------------------------------------------------------------------
RUN cd ${COMFYUI_PATH}/custom_nodes && \
    git clone --depth 1 https://github.com/saganaki22/ComfyUI-FishAudioS2.git && \
    cd ComfyUI-FishAudioS2 && \
    pip install --no-cache-dir -r requirements.txt

# descript-audio-codec y descript-audiotools van aparte con --no-deps a
# propósito (según el README del proyecto): instalarlos con sus deps
# arrastra un pin de protobuf<5 que rompe otros nodos en un entorno
# compartido como este. El propio custom node los auto-instala al primer
# arranque de ComfyUI si no están, pero los dejamos ya listos en el build
# para no pagar ese costo en el primer cold start del serverless.
RUN pip install --no-cache-dir --no-deps descript-audio-codec "descript-audiotools>=0.7.2" && \
    pip install --no-cache-dir flatten-dict importlib-resources julius randomname ffmpy argbind

# Modelo full precision (~24GB, requiere ~24GB VRAM libres según el README
# del proyecto — ver aviso en la conversación sobre VRAM compartida con
# Qwen3.5). Si más adelante hace falta cambiar a la variante FP8
# (drbaph/s2-pro-fp8, ~20GB VRAM), es cuestión de cambiar estas dos líneas.
ENV FISH_MODEL_REPO=fishaudio/s2-pro
ENV FISH_MODEL_DIR=${COMFYUI_PATH}/models/fishaudioS2/s2-pro

RUN mkdir -p ${FISH_MODEL_DIR} && \
    hf download ${FISH_MODEL_REPO} --local-dir ${FISH_MODEL_DIR}

RUN test -d ${FISH_MODEL_DIR} && \
    size_mb=$(du -sm ${FISH_MODEL_DIR} | cut -f1) && \
    echo "Fish Audio S2-Pro descargado: ${size_mb}MB" && \
    [ "$size_mb" -gt 5000 ] || (echo "ERROR: el modelo de Fish Audio no se descargó completo" && exit 1)

WORKDIR ${COMFYUI_PATH}
CMD ["python3", "-u", "handler.py"]
