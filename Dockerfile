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
# Workflow-to-API converter: agrega el endpoint /workflow/convert, que
# convierte un workflow en formato UI (el que exporta ComfyUI con "Save",
# no "Save (API)") al formato API real, usando la misma lógica que el
# botón "Save (API)" del frontend. Lo usamos para generar el
# workflow_api.json definitivo sin adivinar el formato de inputs raros
# como COMFY_DYNAMICCOMBO_V3 (sampling_mode de TextGenerate).
# https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint
# ---------------------------------------------------------------------------
RUN cd ${COMFYUI_PATH}/custom_nodes && \
    git clone --depth 1 https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git

COPY ui_workflow_source.json ${COMFYUI_PATH}/ui_workflow_source.json

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

WORKDIR ${COMFYUI_PATH}
CMD ["python3", "-u", "handler.py"]
