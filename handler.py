"""
RunPod Serverless handler multimodal para ComfyUI:
- Mode "llm": Qwen3.5-4B (CLIPLoader + TextGenerate)
- Mode "tts": Fish Audio S2-Pro TTS
- Mode "voice_clone": Fish Audio S2-Pro Voice Cloning
"""

import base64
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
INPUT_DIR = os.path.join(COMFYUI_PATH, "input")
OUTPUT_DIR = os.path.join(COMFYUI_PATH, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Node IDs del workflow base de LLM (Qwen3.5)
NODE_ID_USER_PROMPT = "31"
NODE_ID_SAMPLER_OPTS = "68"

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


# ---------------------------------------------------------------------------
# Constructores de Workflows Dinámicos (TTS & Voice Clone)
# ---------------------------------------------------------------------------

def _build_llm_workflow(job_input):
    with open(WORKFLOW_PATH, "r") as f:
        wf = json.load(f)

    prompt_text = job_input.get("prompt", "")
    system_prompt = job_input.get("system_prompt", "")
    full_prompt = f"{system_prompt}\n\n{prompt_text}" if system_prompt else prompt_text

    if NODE_ID_USER_PROMPT in wf:
        wf[NODE_ID_USER_PROMPT]["inputs"]["value"] = full_prompt

    if NODE_ID_SAMPLER_OPTS in wf:
        node_inputs = wf[NODE_ID_SAMPLER_OPTS]["inputs"]
        node_inputs["max_length"] = job_input.get("max_length", node_inputs.get("max_length", 1024))
        node_inputs["thinking"] = job_input.get("thinking", node_inputs.get("thinking", False))

        do_sample = job_input.get("sampling_mode", "on") != "off"
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
            node_inputs["sampling_mode.repetition_penalty"] = job_input.get("repetition_penalty", 1.05)
            node_inputs["sampling_mode.seed"] = job_input.get("seed", 0)
        else:
            node_inputs["sampling_mode"] = "off"

    return wf


def _build_tts_workflow(job_input):
    text = job_input.get("text", "")
    language = job_input.get("language", "auto")

    return {
        "1": {
            "class_type": "FishS2TTS",
            "inputs": {
                "text": text,
                "model_path": "s2-pro",
                "language": language,
                "device": "auto",
                "precision": "auto",
                "attention": "auto",
                "chunk_length": 0,
                "max_new_tokens": 0,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "seed": 0,
                "keep_model_loaded": True,
                "compile_model": False,
                "offload_to_cpu": False,
            },
        },
        "2": {
            "class_type": "SaveAudioMP3",
            "inputs": {
                "audio": ["1", 0],
                "filename_prefix": "tts_output",
                "quality": "320k",
            },
        },
    }


def _build_voice_clone_workflow(job_input):
    text = job_input.get("text", "")
    language = job_input.get("language", "auto")
    ref_b64 = job_input.get("reference_audio_b64", "")
    ref_format = job_input.get("reference_audio_format", "ogg")

    # Guardar audio de referencia localmente para LoadAudio
    filename = f"ref_{uuid.uuid4().hex[:8]}.{ref_format}"
    filepath = os.path.join(INPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(ref_b64))

    return {
        "1": {
            "class_type": "LoadAudio",
            "inputs": {
                "audio": filename,
            },
        },
        "2": {
            "class_type": "FishS2VoiceCloneTTS",
            "inputs": {
                "text": text,
                "reference_audio": ["1", 0],
                "model_path": "s2-pro",
                "language": language,
                "device": "auto",
                "precision": "auto",
                "attention": "auto",
                "chunk_length": 0,
                "max_new_tokens": 0,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "seed": 0,
                "keep_model_loaded": True,
                "compile_model": False,
                "offload_to_cpu": False,
            },
        },
        "3": {
            "class_type": "SaveAudioMP3",
            "inputs": {
                "audio": ["2", 0],
                "filename_prefix": "clone_output",
                "quality": "320k",
            },
        },
    }


# ---------------------------------------------------------------------------
# Ejecución y extracción de resultados
# ---------------------------------------------------------------------------

def _queue_prompt(wf):
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": wf, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


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


def _extract_audio_base64(history):
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        if "audio" in node_out:
            audio_info_list = node_out["audio"]
            if audio_info_list and len(audio_info_list) > 0:
                audio_info = audio_info_list[0]
                filename = audio_info.get("filename")
                subfolder = audio_info.get("subfolder", "")
                file_type = audio_info.get("type", "output")

                filepath = os.path.join(OUTPUT_DIR, subfolder, filename) if subfolder else os.path.join(OUTPUT_DIR, filename)

                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode("utf-8")
                    fmt = filename.split(".")[-1] if "." in filename else "mp3"
                    return {"audio_base64": b64_data, "audio_format": fmt}

    raise FileNotFoundError(f"No se encontró el archivo de audio generado en los outputs: {outputs}")


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


# ---------------------------------------------------------------------------
# Handler principal
# ---------------------------------------------------------------------------

def handler(job):
    job_input = job.get("input", {})
    mode = job_input.get("mode", "llm")

    _start_comfyui()

    # Comandos de depuración existentes
    if job_input.get("debug") == "convert_workflow":
        ui_workflow_path = os.path.join(COMFYUI_PATH, "ui_workflow_source.json")
        with open(ui_workflow_path, "r") as f:
            ui_workflow = json.load(f)
        r = requests.post(f"{COMFY_URL}/workflow/convert", json=ui_workflow, timeout=60)
        r.raise_for_status()
        return {"api_workflow": r.json()}

    if job_input.get("debug") == "object_info":
        node_class = job_input.get("node_class", "TextGenerate")
        r = requests.get(f"{COMFY_URL}/object_info/{node_class}", timeout=30)
        r.raise_for_status()
        return {"object_info": r.json()}

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

    # Procesamiento por modos
    try:
        if mode == "tts":
            if not job_input.get("text"):
                return {"error": "Falta el campo 'text' para mode='tts'"}
            wf = _build_tts_workflow(job_input)
            prompt_id = _queue_prompt(wf)
            history = _poll_history(prompt_id)
            return _extract_audio_base64(history)

        elif mode == "voice_clone":
            if not job_input.get("text") or not job_input.get("reference_audio_b64"):
                return {"error": "Faltan campos 'text' o 'reference_audio_b64' para mode='voice_clone'"}
            wf = _build_voice_clone_workflow(job_input)
            prompt_id = _queue_prompt(wf)
            history = _poll_history(prompt_id)
            return _extract_audio_base64(history)

        else:  # Modo "llm" por defecto
            if not job_input.get("prompt"):
                return {"error": "Falta el campo 'prompt' para el LLM"}
            wf = _build_llm_workflow(job_input)
            prompt_id = _queue_prompt(wf)
            history = _poll_history(prompt_id)
            text = _extract_text(history)
            return {"response": text}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})