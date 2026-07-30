"""
RunPod Serverless handler para correr Qwen3.5-4B a través de ComfyUI,
usando los nodos nativos CLIPLoader + TextGenerate (comfy-core).

Input esperado (job["input"]):
{
    "prompt": "texto del usuario",
    "system_prompt": "opcional",
    "max_length": 1024,          # opcional
    "sampling_mode": "on",       # opcional: "on" o "off". Si es "off" se
                                  # ignoran temperature/top_k/etc (greedy).
    "thinking": false,           # opcional
    "use_default_template": true,# opcional
    "temperature": 0.7,          # opcional (solo aplica si sampling_mode="on")
    "top_k": 64,                 # opcional
    "top_p": 0.95,               # opcional
    "min_p": 0.05,               # opcional
    "repetition_penalty": 1.05,  # opcional
    "presence_penalty": 0.0,     # opcional
    "seed": 0,                   # opcional
    "workflow_overrides": {...}  # opcional: para pisar nodos puntuales
}

Nota interna: TextGenerate.sampling_mode es un campo COMFY_DYNAMICCOMBO_V3
(confirmado vía /object_info). El handler arma automáticamente la
estructura anidada {"key": "on"/"off", "inputs": {...}} que esto requiere;
no hace falta que quien llame al endpoint lo sepa.

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

        node_inputs["max_length"] = job_input.get("max_length", node_inputs.get("max_length", 1024))
        node_inputs["thinking"] = job_input.get("thinking", node_inputs.get("thinking", False))
        node_inputs["use_default_template"] = job_input.get(
            "use_default_template", node_inputs.get("use_default_template", True)
        )

        # sampling_mode es un COMFY_DYNAMICCOMBO_V3. Confirmado con el
        # endpoint /workflow/convert (Save-API real): NO es un dict anidado
        # ni claves sueltas -- son claves con notación de punto
        # "sampling_mode.<campo>", más "sampling_mode" como string plano
        # ("on"/"off").
        do_sample = job_input.get("sampling_mode", "on") != "off"

        # Limpiar cualquier resabio de intentos anteriores (dict anidado)
        node_inputs.pop("sampling_mode", None)
        for k in list(node_inputs.keys()):
            if k.startswith("sampling_mode."):
                del node_inputs[k]

        if do_sample:
            node_inputs["sampling_mode"] = "on"
            node_inputs["sampling_mode.temperature"] = job_input.get("temperature", 0.7)
            node_inputs["sampling_mode.top_k"] = job_input.get("top_k", 64)
            node_inputs["sampling_mode.top_p"] = job_input.get("top_p", 0.95)
            node_inputs["sampling_mode.min_p"] = job_input.get("min_p", 0.05)
            node_inputs["sampling_mode.repetition_penalty"] = job_input.get(
                "repetition_penalty", 1.05
            )
            node_inputs["sampling_mode.seed"] = job_input.get("seed", 0)
            node_inputs["sampling_mode.presence_penalty"] = job_input.get(
                "presence_penalty", 0.0
            )
        else:
            node_inputs["sampling_mode"] = "off"

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

    _start_comfyui()

    # Modo debug: convierte el workflow original en formato UI (bundleado en
    # ui_workflow_source.json) al formato API real, usando el endpoint
    # /workflow/convert (mismo código que usa el botón "Save (API)" del
    # frontend). Esto da el JSON garantizado correcto para inputs raros
    # como el sampling_mode (COMFY_DYNAMICCOMBO_V3) de TextGenerate, sin
    # tener que adivinar la estructura a mano.
    # Uso: {"input": {"debug": "convert_workflow"}}
    if job_input.get("debug") == "convert_workflow":
        ui_workflow_path = os.path.join(COMFYUI_PATH, "ui_workflow_source.json")
        with open(ui_workflow_path, "r") as f:
            ui_workflow = json.load(f)
        r = requests.post(f"{COMFY_URL}/workflow/convert", json=ui_workflow, timeout=60)
        r.raise_for_status()
        return {"api_workflow": r.json()}

    # Modo debug: en vez de generar texto, devuelve la definición real del
    # nodo (tal como la ve /object_info de ComfyUI). Sirve para confirmar
    # el formato exacto que espera un input tipo DynamicCombo (como
    # sampling_mode) sin necesitar abrir la UI en un Pod aparte.
    # Uso: {"input": {"debug": "object_info", "node_class": "TextGenerate"}}
    if job_input.get("debug") == "object_info":
        node_class = job_input.get("node_class", "TextGenerate")
        r = requests.get(f"{COMFY_URL}/object_info/{node_class}", timeout=30)
        r.raise_for_status()
        return {"object_info": r.json()}

    # Modo debug: busca nodos por substring en su class_type o display_name.
    # Útil cuando conocemos el nombre "bonito" que se ve en la UI (ej. "Fish
    # S2 TTS") pero no el class_type interno que hay que usar en el JSON de
    # /prompt. Devuelve solo nombre + tipo de cada input para no mandar
    # megabytes de object_info completo.
    # Uso: {"input": {"debug": "search_nodes", "query": "fish"}}
    if job_input.get("debug") == "search_nodes":
        query = job_input.get("query", "").lower()
        r = requests.get(f"{COMFY_URL}/object_info", timeout=60)
        r.raise_for_status()
        all_nodes = r.json()
        matches = {}
        for class_type, info in all_nodes.items():
            display_name = info.get("display_name") or ""
            if query in class_type.lower() or query in display_name.lower():
                inputs = info.get("input", {})
                matches[class_type] = {
                    "display_name": display_name,
                    "required": {k: v[0] for k, v in inputs.get("required", {}).items()},
                    "optional": {k: v[0] for k, v in inputs.get("optional", {}).items()},
                    "output": info.get("output"),
                    "output_name": info.get("output_name"),
                }
        return {"matches": matches}

    if not job_input.get("prompt"):
        return {"error": "Falta el campo 'prompt' en el input"}

    wf = _build_prompt(job_input)
    prompt_id, _ = _queue_prompt(wf)
    history = _poll_history(prompt_id)
    text = _extract_text(history)
    return {"response": text}


runpod.serverless.start({"handler": handler})