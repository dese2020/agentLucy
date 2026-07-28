"""
RunPod Serverless handler para correr un LLM (Qwen3.5-4B-Deckard-HERETIC)
a través de ComfyUI (usando comfyui_LLM_party como backend de chat).

Input esperado (job["input"]):
{
    "prompt": "texto del usuario",
    "system_prompt": "opcional",
    "max_tokens": 512,           # opcional
    "temperature": 0.8,          # opcional
    "workflow_overrides": {...}  # opcional: para pisar nodos puntuales
}

Output:
{
    "response": "texto generado por el modelo"
}
"""

import json
import os
import subprocess
import time
import uuid

import requests
import runpod

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/opt/ComfyUI")
WORKFLOW_PATH = os.path.join(COMFYUI_PATH, "workflow_api.json")

# IDs de los nodos dentro del workflow_api.json, tomados de tu grafo real
# (ep38_local_llm_text_generate.json), rama de solo texto:
#   61 = CLIPLoader (carga el modelo)
#   31 = PrimitiveStringMultiline (prompt del usuario)
#   68 = TextGenerate (genera el texto)
#   11 = PreviewAny (nodo de salida que aparece en /history)
# Si volvés a exportar el workflow desde ComfyUI, verificá que estos IDs
# no hayan cambiado.
NODE_ID_USER_PROMPT = "31"
NODE_ID_MODEL_LOADER = "61"
NODE_ID_SAMPLER_OPTS = "68"
NODE_ID_OUTPUT = "11"

_comfy_process = None


def _wait_for_server(timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("ComfyUI no arrancó a tiempo")


def _start_comfyui():
    global _comfy_process
    if _comfy_process is not None and _comfy_process.poll() is None:
        return
    _comfy_process = subprocess.Popen(
        [
            "python3",
            os.path.join(COMFYUI_PATH, "main.py"),
            "--listen", COMFY_HOST,
            "--port", str(COMFY_PORT),
            "--disable-auto-launch",
        ],
        cwd=COMFYUI_PATH,
    )
    _wait_for_server()


def _load_workflow():
    with open(WORKFLOW_PATH, "r") as f:
        return json.load(f)


def _build_prompt(job_input):
    wf = _load_workflow()

    prompt_text = job_input.get("prompt", "")
    system_prompt = job_input.get("system_prompt", "")
    max_tokens = job_input.get("max_tokens", 512)
    temperature = job_input.get("temperature", 0.8)

    # Este workflow no tiene un nodo de system prompt separado (el modelo de
    # DavidAU "no necesita system prompt" según su model card). Si querés uno,
    # lo simple es concatenarlo delante del prompt del usuario.
    full_prompt = f"{system_prompt}\n\n{prompt_text}" if system_prompt else prompt_text

    if NODE_ID_USER_PROMPT in wf:
        # OJO: "value" es un supuesto -- confirmá el nombre real del widget
        # de PrimitiveStringMultiline exportando el workflow con
        # Workflow > Export (API) y mirando la clave dentro de "inputs".
        wf[NODE_ID_USER_PROMPT]["inputs"]["value"] = full_prompt
    if NODE_ID_SAMPLER_OPTS in wf:
        # Ídem: "max_new_tokens"/"temperature" son los nombres más probables
        # para los widgets de TextGenerate, pero confirmalos contra el
        # workflow_api.json real antes de confiar en esto en producción.
        wf[NODE_ID_SAMPLER_OPTS]["inputs"]["max_new_tokens"] = max_tokens
        wf[NODE_ID_SAMPLER_OPTS]["inputs"]["temperature"] = temperature

    # Overrides manuales opcionales, por si querés pisar cualquier campo directo
    for node_id, fields in job_input.get("workflow_overrides", {}).items():
        wf.setdefault(node_id, {}).setdefault("inputs", {}).update(fields)

    return wf


def _queue_prompt(wf):
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": wf, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"], client_id


def _poll_history(prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if prompt_id in data:
            return data[prompt_id]
        time.sleep(1)
    raise TimeoutError("Timeout esperando resultado de ComfyUI")


def _extract_text(history):
    """
    Recorre los outputs del history buscando el primer campo de texto.
    Dependiendo del node pack, la clave puede ser 'text', 'string', etc.
    Ajustá según lo que devuelva tu nodo de salida.
    """
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        for key in ("text", "string", "output"):
            if key in node_out:
                value = node_out[key]
                if isinstance(value, list):
                    return "\n".join(str(v) for v in value)
                return str(value)
    return json.dumps(outputs)


def handler(job):
    job_input = job.get("input", {})
    if not job_input.get("prompt"):
        return {"error": "Falta el campo 'prompt' en el input"}

    _start_comfyui()
    wf = _build_prompt(job_input)
    prompt_id, _ = _queue_prompt(wf)
    history = _poll_history(prompt_id)
    text = _extract_text(history)
    return {"response": text}


runpod.serverless.start({"handler": handler})
