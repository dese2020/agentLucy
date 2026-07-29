"""
RunPod Serverless handler para correr un LLM (Qwen3.5-4B-Deckard-HERETIC)
a través de ComfyUI (usando comfyui_LLM_party como backend de chat).

Input esperado (job["input"]):
{
    "prompt": "texto del usuario",
    "system_prompt": "opcional",
    "max_length": 1024,          # opcional
    "sampling_mode": "on",       # opcional ("on", "off")
    "thinking": false,           # opcional
    "use_default_template": true,# opcional
    "temperature": 0.7,          # opcional
    "top_k": 64,                 # opcional
    "top_p": 0.95,               # opcional
    "min_p": 0.05,               # opcional
    "repetition_penalty": 1.05,  # opcional
    "presence_penalty": 0.0,     # opcional
    "seed": 0,                   # opcional
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

# IDs de los nodos dentro del workflow_api.json:
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

    # Concatenar el system_prompt si está presente
    full_prompt = f"{system_prompt}\n\n{prompt_text}" if system_prompt else prompt_text

    # 1. Configurar nodo de Prompt
    if NODE_ID_USER_PROMPT in wf:
        wf[NODE_ID_USER_PROMPT]["inputs"]["value"] = full_prompt

    # 2. Configurar todos los parámetros del nodo TextGenerate (68)
    if NODE_ID_SAMPLER_OPTS in wf:
        node_inputs = wf[NODE_ID_SAMPLER_OPTS]["inputs"]
        
        # Mapeo de parámetros con sus valores por defecto
        node_inputs["max_length"] = job_input.get("max_length", node_inputs.get("max_length", 1024))
        node_inputs["sampling_mode"] = job_input.get("sampling_mode", node_inputs.get("sampling_mode", "on"))
        node_inputs["thinking"] = job_input.get("thinking", node_inputs.get("thinking", False))
        node_inputs["use_default_template"] = job_input.get("use_default_template", node_inputs.get("use_default_template", True))
        node_inputs["temperature"] = job_input.get("temperature", node_inputs.get("temperature", 0.7))
        node_inputs["top_k"] = job_input.get("top_k", node_inputs.get("top_k", 64))
        node_inputs["top_p"] = job_input.get("top_p", node_inputs.get("top_p", 0.95))
        node_inputs["min_p"] = job_input.get("min_p", node_inputs.get("min_p", 0.05))
        node_inputs["repetition_penalty"] = job_input.get("repetition_penalty", node_inputs.get("repetition_penalty", 1.05))
        node_inputs["presence_penalty"] = job_input.get("presence_penalty", node_inputs.get("presence_penalty", 0.0))
        node_inputs["seed"] = job_input.get("seed", node_inputs.get("seed", 0))

    # Overrides manuales opcionales
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