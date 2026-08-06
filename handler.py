"""
RunPod Serverless handler multimodal para ComfyUI:
- Mode "llm": Qwen3.5-4B Heretic Uncensored (CLIPLoader + TextGenerate)
- Mode "tts": Fish Audio S2-Pro TTS
- Mode "voice_clone": Fish Audio S2-Pro Voice Cloning
- Debug utilities: convert_workflow, object_info, search_nodes
"""

import base64
import json
import os
import subprocess
import time
import uuid

import requests
import runpod
from faster_whisper import WhisperModel

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/opt/ComfyUI")
WORKFLOW_PATH = os.path.join(COMFYUI_PATH, "workflow_api.json")
INPUT_DIR = os.path.join(COMFYUI_PATH, "input")
OUTPUT_DIR = os.path.join(COMFYUI_PATH, "output")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Node IDs del workflow base de LLM
NODE_ID_USER_PROMPT = "31"
NODE_ID_SAMPLER_OPTS = "68"

_comfy_process = None

# --- ASR (Whisper) para autotranscribir el audio de referencia ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
WHISPER_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR", "/opt/whisper_cache")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")  # "cuda" si tienes VRAM libre
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=WHISPER_CACHE_DIR,
        )
    return _whisper_model


def _transcribe_audio(filepath, language=None):
    """Transcribe un archivo de audio. Si language es None o 'auto', detecta el idioma."""
    model = _get_whisper_model()
    lang_arg = None if (not language or language == "auto") else language
    segments, info = model.transcribe(filepath, language=lang_arg, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip(), info.language


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
        node_inputs["thinking"] = job_input.get("thinking", node_inputs.get("thinking", True))
        node_inputs["use_default_template"] = job_input.get(
            "use_default_template", node_inputs.get("use_default_template", True)
        )

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
            node_inputs["sampling_mode.presence_penalty"] = job_input.get("presence_penalty", 0.0)
            node_inputs["sampling_mode.seed"] = job_input.get("seed", 0)
        else:
            node_inputs["sampling_mode"] = "off"

    # Overrides opcionales
    for node_id, fields in job_input.get("workflow_overrides", {}).items():
        wf.setdefault(node_id, {}).setdefault("inputs", {}).update(fields)

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
                "chunk_length": 200,
                "max_new_tokens": 1024,
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
    ref_text = job_input.get("reference_text", "")  # Transcripción de la voz de referencia

    filename = f"ref_{uuid.uuid4().hex[:8]}.{ref_format}"
    filepath = os.path.join(INPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(ref_b64))

    if not ref_text:
        detected_text, detected_lang = _transcribe_audio(filepath, language=language)
        ref_text = detected_text
        # Si el usuario dejó language="auto", usamos el idioma detectado por Whisper
        # para el TTS también (mejor que dejarlo en "auto" para Fish Audio).
        if language == "auto" and detected_lang:
            language = detected_lang

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
                "reference_text": ref_text,  # Se pasa la transcripción de referencia
                "model_path": "s2-pro",
                "language": language,
                "device": "auto",
                "precision": "auto",
                "attention": "auto",
                "chunk_length": 200,
                "max_new_tokens": 1024,
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

                if file_type == "temp":
                    dir_path = os.path.join(COMFYUI_PATH, "temp", subfolder) if subfolder else os.path.join(COMFYUI_PATH, "temp")
                else:
                    dir_path = os.path.join(OUTPUT_DIR, subfolder) if subfolder else OUTPUT_DIR

                filepath = os.path.join(dir_path, filename)

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


def handler(job):
    job_input = job.get("input", {})
    mode = job_input.get("mode", "llm")

    _start_comfyui()

    # --- Herramientas de Debug ---
    if job_input.get("debug") == "convert_workflow":
        ui_workflow = None

        # 1. Verificar si viene codificado en Base64
        if job_input.get("ui_workflow_b64"):
            try:
                b64_bytes = job_input["ui_workflow_b64"].encode("utf-8")
                json_bytes = base64.b64decode(b64_bytes)
                ui_workflow = json.loads(json_bytes.decode("utf-8"))
            except Exception as err:
                return {"error": f"Error al decodificar ui_workflow_b64: {str(err)}"}

        # 2. Si no viene en Base64, intentar obtener el JSON directo
        if not ui_workflow:
            ui_workflow = job_input.get("ui_workflow") or job_input.get("workflow")

        # 3. Fallback: buscar el archivo local por defecto
        if not ui_workflow:
            ui_workflow_path = os.path.join(COMFYUI_PATH, "ui_workflow_source.json")
            if os.path.exists(ui_workflow_path):
                with open(ui_workflow_path, "r") as f:
                    ui_workflow = json.load(f)
            else:
                return {"error": "No se proporcionó 'ui_workflow_b64', 'ui_workflow' ni existe el archivo local por defecto."}

        # 4. Enviar a ComfyUI para su conversión
        try:
            r = requests.post(f"{COMFY_URL}/workflow/convert", json=ui_workflow, timeout=60)
            r.raise_for_status()
            return {"api_workflow": r.json()}
        except Exception as e:
            return {"error": f"Fallo al convertir el workflow en ComfyUI: {str(e)}"}

    if job_input.get("debug") == "transcribe":
        ref_b64 = job_input.get("reference_audio_b64", "")
        ref_format = job_input.get("reference_audio_format", "ogg")
        language = job_input.get("language", "auto")
        if not ref_b64:
            return {"error": "Falta 'reference_audio_b64' para debug='transcribe'"}
        filename = f"debug_{uuid.uuid4().hex[:8]}.{ref_format}"
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(ref_b64))
        try:
            text, detected_lang = _transcribe_audio(filepath, language=language)
            return {"transcription": text, "detected_language": detected_lang}
        except Exception as e:
            return {"error": f"Fallo al transcribir: {str(e)}"}

    if job_input.get("debug") == "object_info":
        node_class = job_input.get("node_class", "TextGenerate")
        r = requests.get(f"{COMFY_URL}/object_info/{node_class}", timeout=30)
        r.raise_for_status()
        return {"object_info": r.json()}

    if job_input.get("debug") == "search_nodes":
        search_term = job_input.get("query", "").lower()
        r = requests.get(f"{COMFY_URL}/object_info", timeout=30)
        r.raise_for_status()
        all_nodes = r.json()
        
        matches = {}
        for node_name, details in all_nodes.items():
            display_name = details.get("display_name", "").lower()
            category = details.get("category", "").lower()
            if search_term in node_name.lower() or search_term in display_name or search_term in category:
                matches[node_name] = {
                    "display_name": details.get("display_name"),
                    "category": details.get("category"),
                    "input_required": list(details.get("input", {}).get("required", {}).keys()),
                    "input_optional": list(details.get("input", {}).get("optional", {}).keys()),
                    "output": details.get("output"),
                }
        return {"search_query": search_term, "count": len(matches), "matches": matches}

    # --- Modos de Producción ---
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
            # reference_text ahora es opcional: si no se envía, se transcribe
            # automáticamente el audio de referencia con Whisper.
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