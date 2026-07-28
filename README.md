# ComfyUI + Qwen3.5-4B en RunPod Serverless + Bot de Telegram

## 1. Cosas que tenés que decidir/ajustar antes de buildear

- **Node pack de LLM en ComfyUI**: el Dockerfile clona
  `comfyui_LLM_party` como ejemplo. Si vos ya usás otro paquete
  (ComfyUI-IF_AI_tools, ComfyUI-Ollama, etc.) cambiá esa línea y
  ajustá `workflow_api.json` con los nombres reales de `class_type`
  e `inputs` de tus nodos (abrí tu workflow en ComfyUI ->
  Workflow -> Export (API) para sacar los IDs correctos).
- **IDs de nodos en `handler.py`**: `NODE_ID_USER_PROMPT`,
  `NODE_ID_SYSTEM_PROMPT`, etc. tienen que matchear los IDs reales
  del `workflow_api.json` exportado.
- **Formato de salida del nodo final**: `_extract_text()` asume que
  el output trae una clave `text`/`string`/`output`. Revisalo con un
  `print(history)` la primera vez.

## 2. Build & push de la imagen

```bash
docker build \
  --build-arg HF_TOKEN=hf_xxxxxxxxxxxxxxxx \
  -t tuusuario/comfyui-qwen-heretic:1.0 .

docker push tuusuario/comfyui-qwen-heretic:1.0
```

> El modelo se descarga **en build time** con `hf download`, así el
> cold start del worker no tiene que bajar ~8GB cada vez. Si preferís
> bajarlo en runtime a un Network Volume en vez de meterlo en la
> imagen (imagen más liviana, pero cold start más lento la primera
> vez), decime y te paso esa variante.

## 3. Deploy en RunPod Serverless

1. Console de RunPod -> Serverless -> New Endpoint.
2. Container Image: `tuusuario/comfyui-qwen-heretic:1.0`.
3. GPU: elegí una con VRAM suficiente para un modelo de 4B (con
   16GB sobra tranquilo, incluso en fp16).
4. Container Disk: ~25-30GB (el modelo + ComfyUI + deps).
5. Guardá el **Endpoint ID** y tu **RunPod API Key**.

## 4. Bot de Telegram

```bash
pip install -r requirements-bot.txt

export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
export RUNPOD_API_KEY="rpa_xxxxxxxxxxxx"
export RUNPOD_ENDPOINT_ID="tu-endpoint-id"

python telegram_bot.py
```

El bot:
- Usa `/runsync` (bloqueante) y si tarda más que el timeout de
  RunPod, cae automáticamente a polling con `/status/{id}`.
- Trocea respuestas largas en bloques de 4000 caracteres (límite de
  Telegram es 4096).
- `/start` y `/reset` limpian el contexto en memoria (no hay
  persistencia entre reinicios del bot; si necesitás memoria
  persistente por usuario, lo agregamos con sqlite o redis).

## 5. Notas sobre el modelo

Usamos `Comfy-Org/Qwen3.5` (el `qwen3.5_4b_bf16.safetensors` que
tu workflow ya referencia en el `CLIPLoader`), en vez del finetune de
DavidAU. Ventaja: viene empaquetado en el formato exacto que
`CLIPLoader` espera, sin necesidad de fusionar shards ni adivinar
compatibilidad de formato. Si más adelante querés el finetune
uncensored de DavidAU, va a hacer falta convertirlo a ese mismo
formato single-file (o confirmar si `CLIPLoader` acepta una carpeta
HF Transformers estándar), y probarlo antes en un Pod normal de
RunPod antes de meterlo en el serverless.
