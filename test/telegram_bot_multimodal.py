"""
Bot de Telegram multimodal sobre dos endpoints serverless de RunPod:

  - Endpoint TTS  (handler_tts.py): Whisper (STT) + Qwen (LLM) + Fish Audio (TTS / voice clone)
  - Endpoint MNX  (handler_mnx.py): MiniMax H3 ref2va (video + audio)

Entrada: texto o audio (voice / audio file).
Salida segun el modo del usuario (/mode):
  - audio : Whisper -> Qwen -> Fish Audio (voz por defecto o clonada)       -> nota de voz
  - texto : Whisper -> Qwen                                                 -> texto
  - video : Whisper -> Qwen (guionista) -> MiniMax H3 ref2va (imagen + voz) -> video con audio

Configuracion por usuario (chat_id), persistida en disco:
  voices/<chat_id>.json     voz de referencia (clonacion Fish Audio y referencia de voz en H3)
  images/<chat_id>.img      imagen de referencia del personaje (image_1 de ref2va)
  settings/<chat_id>.json   modo, duracion del video, system prompt

Requisitos:
    pip install "python-telegram-bot==21.*" requests python-dotenv
    ffmpeg instalado en la maquina del bot (para normalizar el audio de referencia de H3)

Variables de entorno (.env):
    TELEGRAM_BOT_TOKEN=...
    RUNPOD_API_KEY=...
    RUNPOD_ENDPOINT_ID=...          # endpoint TTS/LLM/Whisper
    RUNPOD_VIDEO_ENDPOINT_ID=...    # endpoint MiniMax H3
    # opcionales
    VIDEO_WIDTH=864
    VIDEO_HEIGHT=480
    VIDEO_TIMEOUT=1500              # segundos max. esperando un video (incluye cold start)
    REF_AUDIO_MAX_SECONDS=10        # recorte del audio de referencia enviado a H3
    WORKFLOW_REF2VA_PATH=...        # workflow ComfyUI (API) enviado en base64; default: junto al script

Comandos:
    /start  /status
    /mode [audio|texto|video]   cambia el modo (sin argumentos muestra botones)
    /setvoice                   luego envia un audio de referencia
    /resetvoice
    /setimage                   luego envia una foto (o enviala con /setimage como pie de foto)
    /resetimage
    /duration <1-15>            segundos del video
    /system <texto>             persona/system prompt (sin texto lo borra)
"""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time

import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
RUNPOD_VIDEO_ENDPOINT_ID = os.environ.get("RUNPOD_VIDEO_ENDPOINT_ID")  # opcional hasta usar modo video

for name, val in [
    ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
    ("RUNPOD_API_KEY", RUNPOD_API_KEY),
    ("RUNPOD_ENDPOINT_ID", RUNPOD_ENDPOINT_ID),
]:
    if not val:
        raise SystemExit(f"Falta la variable de entorno {name}")

HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", 864))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", 480))
VIDEO_TIMEOUT = int(os.environ.get("VIDEO_TIMEOUT", 1500))
REF_AUDIO_MAX_SECONDS = int(os.environ.get("REF_AUDIO_MAX_SECONDS", 10))

# Workflow de ComfyUI (formato API) que se envia en cada job al endpoint MNX como
# 'workflow_ref2va_b64'. Por defecto: junto a este script.
WORKFLOW_REF2VA_PATH = os.environ.get(
    "WORKFLOW_REF2VA_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_h3_ref2va_api.json")
)

MAX_VIDEO_SECONDS = 15
VIDEO_SECONDS_PER_5S_CLIP = 120  # solo para mostrar un ETA aproximado
VIDEO_WORDS_PER_SEC = 2.0        # palabras habladas por segundo (conservador)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MODES = ("audio", "texto", "video")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
SETTINGS_DIR = os.path.join(BASE_DIR, "settings")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
for d in (VOICES_DIR, IMAGES_DIR, SETTINGS_DIR, TMP_DIR):
    os.makedirs(d, exist_ok=True)

# chat_id -> "voice" | "image"  (esperando el siguiente archivo para configurar)
_awaiting = {}
# chat_ids con un video en proceso
_video_busy = set()


# ---------------------------------------------------------------------------
# RunPod
# ---------------------------------------------------------------------------
def run_job(endpoint_id, payload, timeout=600, sync=True):
    """
    Ejecuta un job y espera el resultado.
    sync=True  -> /runsync (rapido para jobs cortos: whisper, llm, tts)
    sync=False -> /run + polling (para video, no mantiene la conexion abierta)
    Bloqueante: desde handlers async llamar con asyncio.to_thread().
    """
    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    if sync:
        resp = requests.post(f"{base}/runsync", headers=HEADERS, json={"input": payload}, timeout=timeout)
    else:
        resp = requests.post(f"{base}/run", headers=HEADERS, json={"input": payload}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("id")

    start = time.time()
    while data.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout esperando el job {job_id} ({timeout}s)")
        time.sleep(3)
        r = requests.get(f"{base}/status/{job_id}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

    status = data.get("status")
    if status != "COMPLETED":
        raise RuntimeError(f"Job termino con estado {status}: {data}")

    output = data.get("output")
    if output is None:
        raise RuntimeError(f"Respuesta inesperada del endpoint: {data}")
    if isinstance(output, dict) and output.get("error"):
        raise RuntimeError(f"Error del handler: {output['error']}")
    return output


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def b64_to_file(b64_data, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))


# ---------------------------------------------------------------------------
# Whisper / Qwen / Fish Audio (endpoint TTS)
# ---------------------------------------------------------------------------
def transcribe(audio_path, language="auto"):
    ext = os.path.splitext(audio_path)[1].lstrip(".") or "ogg"
    payload = {
        "debug": "transcribe",
        "reference_audio_b64": file_to_b64(audio_path),
        "reference_audio_format": ext,
        "language": language,
    }
    out = run_job(RUNPOD_ENDPOINT_ID, payload)
    return out["transcription"], out.get("detected_language")


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Fish Audio S2-Pro corta en ~1024 tokens de audio (~47 s) -> ~100 palabras reales.
TTS_MAX_WORDS = 350
TTS_MAX_CHARS = 600

# Prompt traducido al inglés para optimizar la salida del LLM
BREVITY_HINT = (
    "Always reply briefly and directly, in a maximum of "
    f"{TTS_MAX_WORDS} words, because your response will be converted to audio "
    "and the audio has a duration limit. Maintain the same language used in the user prompt."
)
TELEGRAM_TEXT_LIMIT = 4000


def strip_thinking(text):
    return THINK_TAG_RE.sub("", text).strip()


def truncate_for_tts(text, max_chars=TTS_MAX_CHARS, max_words=TTS_MAX_WORDS):
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) > max_chars:
        text = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n"):
        idx = text.rfind(sep)
        if idx > max_chars * 0.4:
            text = text[: idx + 1]
            break
    else:
        text = text.rsplit(" ", 1)[0] if " " in text else text
    return text.strip()


def ask_llm(prompt, system_prompt=None):
    payload = {"mode": "llm", "prompt": prompt, "thinking": False}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    out = run_job(RUNPOD_ENDPOINT_ID, payload)
    return strip_thinking(out["response"])


def text_to_speech(text, language="auto"):
    out = run_job(RUNPOD_ENDPOINT_ID, {"mode": "tts", "text": text, "language": language})
    return out["audio_base64"], out["audio_format"]


def text_to_speech_cloned(text, voice_cfg, language="auto"):
    payload = {
        "mode": "voice_clone",
        "text": text,
        "language": language,
        "reference_audio_b64": voice_cfg["reference_audio_b64"],
        "reference_audio_format": voice_cfg["reference_audio_format"],
    }
    if voice_cfg.get("reference_text"):
        payload["reference_text"] = voice_cfg["reference_text"]
    out = run_job(RUNPOD_ENDPOINT_ID, payload)
    return out["audio_base64"], out["audio_format"]


# ---------------------------------------------------------------------------
# Video: guionista (Qwen) + MiniMax H3 ref2va (endpoint MNX)
# ---------------------------------------------------------------------------
def build_director_system(duration, persona=""):
    max_words = max(4, int(duration * VIDEO_WORDS_PER_SEC))
    persona_block = f"\nCharacter personality: {persona}\n" if persona else ""
    # Prompts de dirección traducidos al inglés para mejor comprensión del modelo
    system = (
        f"You are the screenwriter for a {duration}-second video clip starring "
        "the person in image 1, who responds to the user's message by speaking to the camera.\n"
        f"{persona_block}"
        "Respond ONLY with these two lines, without explanations or extra text:\n"
        f"SPEECH: <what the person says out loud, maximum {max_words} words, "
        "in the same language as the user, without emojis or quotation marks>\n"
        "SCENE: <one or two sentences describing location, action, gestures, camera, and environment; "
        "start with 'The person in image 1'>"
    )
    return system, max_words


DEFAULT_SCENE = "The person in image 1 looks at the camera and speaks naturally, medium shot, soft lighting."


def parse_director_output(raw, max_words):
    """Extrae SPEECH y SCENE de la salida del LLM, con fallback si no respeto el formato."""
    habla_m = re.search(r"(?:SPEECH|HABLA)\s*:\s*(.+?)(?=\n\s*(?:SCENE|ESCENA)\s*:|\Z)", raw, re.DOTALL | re.IGNORECASE)
    escena_m = re.search(r"(?:SCENE|ESCENA)\s*:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)

    if habla_m:
        habla = habla_m.group(1)
    else:
        # El modelo ignoro el formato: usamos lo que haya como texto hablado.
        habla = re.split(r"\n\s*(?:SCENE|ESCENA)\s*:", raw, flags=re.IGNORECASE)[0]
    escena = escena_m.group(1).strip() if escena_m else DEFAULT_SCENE

    habla = " ".join(habla.replace('"', "").replace("“", "").replace("”", "").split())
    words = habla.split()
    if len(words) > max_words:
        habla = " ".join(words[:max_words])
    escena = " ".join(escena.split())
    return habla, escena


def build_video_prompt(habla, escena):
    return f'{escena} The person in image 1 says: "{habla}"'


def ref_audio_to_wav_b64(voice_cfg):
    """
    El handler MNX guarda el audio de referencia siempre como .wav, asi que lo
    normalizamos aqui (wav mono, recortado a REF_AUDIO_MAX_SECONDS).
    """
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg no encontrado: se envia el audio de referencia sin convertir")
        return voice_cfg["reference_audio_b64"]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, f"in.{voice_cfg.get('reference_audio_format', 'ogg')}")
        dst = os.path.join(d, "out.wav")
        b64_to_file(voice_cfg["reference_audio_b64"], src)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-t", str(REF_AUDIO_MAX_SECONDS), "-ac", "1", dst],
            check=True,
        )
        return file_to_b64(dst)


def load_workflow_b64(path=WORKFLOW_REF2VA_PATH):
    """
    Lee el workflow desde disco en cada job (los cambios se aplican sin reiniciar el bot),
    valida que tenga los nodos que el handler MNX necesita y lo devuelve en base64.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro el workflow en {path} (ajusta WORKFLOW_REF2VA_PATH)")
    with open(path, "rb") as f:
        raw = f.read()
    wf = json.loads(raw.decode("utf-8"))
    wf = wf.get("prompt", wf)
    missing = [n for n in ("123", "124", "129", "136") if n not in wf]
    if missing:
        raise ValueError(
            f"El workflow {os.path.basename(path)} no esta en formato API o le faltan los nodos {missing}"
        )
    return base64.b64encode(raw).decode("utf-8")


def generate_video(prompt, image_bytes, duration, voice_cfg=None):
    payload = {
        "mode": "ref2va",
        "prompt": prompt,
        "image_1": base64.b64encode(image_bytes).decode("utf-8"),
        "duration": duration,
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "workflow_ref2va_b64": load_workflow_b64(),
    }
    if voice_cfg:
        payload["audio"] = ref_audio_to_wav_b64(voice_cfg)
    out = run_job(RUNPOD_VIDEO_ENDPOINT_ID, payload, timeout=VIDEO_TIMEOUT, sync=False)
    return out["video_base64"]


# ---------------------------------------------------------------------------
# Persistencia por usuario
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {"mode": "audio", "duration": 5, "system_prompt": ""}


def settings_path(chat_id):
    return os.path.join(SETTINGS_DIR, f"{chat_id}.json")


def load_settings(chat_id):
    cfg = dict(DEFAULT_SETTINGS)
    path = settings_path(chat_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg.update(json.load(f))
    return cfg


def save_settings(chat_id, **changes):
    cfg = load_settings(chat_id)
    cfg.update(changes)
    with open(settings_path(chat_id), "w") as f:
        json.dump(cfg, f)
    return cfg


def voice_path(chat_id):
    return os.path.join(VOICES_DIR, f"{chat_id}.json")


def load_voice_config(chat_id):
    path = voice_path(chat_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_voice_config(chat_id, ref_audio_path, language="auto"):
    ext = os.path.splitext(ref_audio_path)[1].lstrip(".") or "ogg"
    ref_b64 = file_to_b64(ref_audio_path)
    ref_text, detected_lang = transcribe(ref_audio_path, language=language)
    cfg = {
        "reference_audio_b64": ref_b64,
        "reference_audio_format": ext,
        "reference_text": ref_text,
        "language": detected_lang if language == "auto" else language,
    }
    with open(voice_path(chat_id), "w") as f:
        json.dump(cfg, f)
    return ref_text


def reset_voice_config(chat_id):
    if os.path.exists(voice_path(chat_id)):
        os.remove(voice_path(chat_id))


def image_path(chat_id):
    return os.path.join(IMAGES_DIR, f"{chat_id}.img")


def load_image(chat_id):
    path = image_path(chat_id)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None


# ---------------------------------------------------------------------------
# Reintentos para llamadas a Telegram (timeouts de red transitorios)
# ---------------------------------------------------------------------------
async def tg_retry(make_call, retries=4, base_delay=2):
    """Ejecuta make_call() (corrutina sin argumentos) reintentando ante TimedOut/NetworkError."""
    for attempt in range(1, retries + 1):
        try:
            return await make_call()
        except (TimedOut, NetworkError) as e:
            if attempt == retries:
                raise
            delay = base_delay * attempt
            log.warning("Telegram fallo (%s), reintento %d/%d en %ds", type(e).__name__, attempt, retries, delay)
            await asyncio.sleep(delay)


async def safe_reply(message, text, **kwargs):
    return await tg_retry(lambda: message.reply_text(text, **kwargs))


# ---------------------------------------------------------------------------
# Respuestas por modo
# ---------------------------------------------------------------------------
async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_text: str):
    """Despacha segun el modo del usuario."""
    chat_id = update.effective_chat.id
    mode = load_settings(chat_id)["mode"]
    if mode == "texto":
        await reply_with_text(update, context, prompt_text)
    elif mode == "video":
        await reply_with_video(update, context, prompt_text)
    else:
        await reply_with_audio(update, context, prompt_text)


async def reply_with_text(update, context, prompt_text):
    chat_id = update.effective_chat.id
    message = update.message
    await message.reply_text("Pensando...")
    system_prompt = load_settings(chat_id)["system_prompt"] or None
    response_text = await asyncio.to_thread(ask_llm, prompt_text, system_prompt)
    await message.reply_text(response_text[:TELEGRAM_TEXT_LIMIT] or "(respuesta vacia)")


async def reply_with_audio(update, context, prompt_text):
    chat_id = update.effective_chat.id
    message = update.message

    await message.reply_text("Pensando...")
    persona = load_settings(chat_id)["system_prompt"]
    system_prompt = f"{persona}\n\n{BREVITY_HINT}" if persona else BREVITY_HINT
    response_text = await asyncio.to_thread(ask_llm, prompt_text, system_prompt)

    tts_text = truncate_for_tts(response_text)
    await message.reply_text(
        response_text if tts_text == response_text
        else f"{tts_text}\n\n(respuesta recortada para el audio)"
    )

    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
    await message.reply_text("Generando audio...")

    voice_cfg = load_voice_config(chat_id)
    if voice_cfg:
        audio_b64, audio_fmt = await asyncio.to_thread(text_to_speech_cloned, tts_text, voice_cfg)
    else:
        audio_b64, audio_fmt = await asyncio.to_thread(text_to_speech, tts_text)

    out_path = os.path.join(TMP_DIR, f"{chat_id}_out.{audio_fmt}")
    b64_to_file(audio_b64, out_path)
    try:
        with open(out_path, "rb") as f:
            if audio_fmt in ("ogg", "oga"):
                await context.bot.send_voice(chat_id=chat_id, voice=f)
            else:
                await context.bot.send_audio(chat_id=chat_id, audio=f)
    finally:
        os.remove(out_path)


async def reply_with_video(update, context, prompt_text):
    """Valida y lanza el job de video en segundo plano para no bloquear el bot."""
    chat_id = update.effective_chat.id
    message = update.message

    if not RUNPOD_VIDEO_ENDPOINT_ID:
        await message.reply_text("Falta RUNPOD_VIDEO_ENDPOINT_ID en el .env para usar el modo video.")
        return
    image_bytes = load_image(chat_id)
    if image_bytes is None:
        await message.reply_text("Para el modo video necesito una imagen de referencia. Usa /setimage y envia una foto.")
        return
    if chat_id in _video_busy:
        await message.reply_text("Todavia estoy generando tu video anterior, espera a que termine.")
        return

    _video_busy.add(chat_id)
    context.application.create_task(
        _video_job(context, chat_id, message, prompt_text, image_bytes), update=update
    )


async def _video_job(context, chat_id, message, prompt_text, image_bytes):
    out_path = os.path.join(TMP_DIR, f"{chat_id}_out.mp4")
    try:
        cfg = load_settings(chat_id)
        duration = cfg["duration"]

        await safe_reply(message, "Escribiendo el guion...")
        system, max_words = build_director_system(duration, cfg["system_prompt"])
        director_prompt = f"{system}\n\nUser message: {prompt_text}"
        raw = await asyncio.to_thread(ask_llm, director_prompt)
        habla, escena = parse_director_output(raw, max_words)
        video_prompt = build_video_prompt(habla, escena)

        voice_cfg = load_voice_config(chat_id)
        eta = int(duration / 5 * VIDEO_SECONDS_PER_5S_CLIP)
        await safe_reply(message, 
            f"Prompt para H3:\n{video_prompt}\n\n"
            f"Generando video de {duration}s"
            f"{' con tu voz de referencia' if voice_cfg else ''} (~{eta}s, puede ser mas si hay cold start)..."
        )
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action="upload_video")
        except (TimedOut, NetworkError):
            pass  # cosmetico, no vale la pena reintentar

        t0 = time.time()
        video_b64 = await asyncio.to_thread(generate_video, video_prompt, image_bytes, duration, voice_cfg)
        elapsed = int(time.time() - t0)

        b64_to_file(video_b64, out_path)

        async def send():
            with open(out_path, "rb") as f:
                await context.bot.send_video(
                    chat_id=chat_id,
                    video=f,
                    caption=f"{habla}\n\n({elapsed}s)"[:1000],
                    width=VIDEO_WIDTH,
                    height=VIDEO_HEIGHT,
                    supports_streaming=True,
                    read_timeout=120,
                    write_timeout=120,
                    connect_timeout=30,
                )

        await tg_retry(send)
    except Exception as e:
        log.exception("Error generando video")
        try:
            await safe_reply(message, f"Ups, fallo la generacion del video: {e}")
        except Exception:
            log.error("Tampoco se pudo avisar el error al usuario (red caida hacia Telegram)")
    finally:
        _video_busy.discard(chat_id)
        if os.path.exists(out_path):
            os.remove(out_path)


# ---------------------------------------------------------------------------
# Handlers de Telegram
# ---------------------------------------------------------------------------
def mode_keyboard(current):
    row = [
        InlineKeyboardButton(("• " if m == current else "") + m, callback_data=f"mode:{m}")
        for m in MODES
    ]
    return InlineKeyboardMarkup([row])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Enviame texto o audio y te respondo segun el modo activo.\n\n"
        "/mode - audio, texto o video\n"
        "/setvoice - audio de referencia para clonar voz (audio y video)\n"
        "/setimage - imagen del personaje para el modo video\n"
        "/duration <1-15> - segundos del video\n"
        "/system <texto> - personalidad / system prompt\n"
        "/status - ver tu configuracion"
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cfg = load_settings(chat_id)
    await update.message.reply_text(
        f"Modo: {cfg['mode']}\n"
        f"Duracion de video: {cfg['duration']}s\n"
        f"Voz de referencia: {'si' if load_voice_config(chat_id) else 'no (voz por defecto)'}\n"
        f"Imagen de referencia: {'si' if os.path.exists(image_path(chat_id)) else 'no'}\n"
        f"System prompt: {cfg['system_prompt'] or '(ninguno)'}"
    )


async def mode_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    arg = context.args[0].lower() if context.args else None
    if arg == "text":
        arg = "texto"
    if arg in MODES:
        save_settings(chat_id, mode=arg)
        await update.message.reply_text(f"Modo cambiado a: {arg}")
        return
    await update.message.reply_text(
        "Elige el modo de respuesta:", reply_markup=mode_keyboard(load_settings(chat_id)["mode"])
    )


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    new_mode = query.data.split(":", 1)[1]
    if new_mode in MODES:
        save_settings(query.message.chat_id, mode=new_mode)
        await query.edit_message_text(f"Modo: {new_mode}", reply_markup=mode_keyboard(new_mode))


async def setvoice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _awaiting[update.effective_chat.id] = "voice"
    await update.message.reply_text("Ok, ahora envia el audio que quieres usar como referencia de voz.")


async def resetvoice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_voice_config(update.effective_chat.id)
    await update.message.reply_text("Listo, vuelvo a la voz por defecto.")


async def setimage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _awaiting[update.effective_chat.id] = "image"
    await update.message.reply_text("Ok, ahora envia la foto del personaje (ref2va usa esta imagen como 'imagen 1').")


async def resetimage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if os.path.exists(image_path(update.effective_chat.id)):
        os.remove(image_path(update.effective_chat.id))
    await update.message.reply_text("Imagen de referencia borrada.")


async def duration_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        value = int(context.args[0])
        if not 1 <= value <= MAX_VIDEO_SECONDS:
            raise ValueError
    except (IndexError, ValueError):
        await update.message.reply_text(f"Uso: /duration <1-{MAX_VIDEO_SECONDS}>  (actual: {load_settings(chat_id)['duration']}s)")
        return
    save_settings(chat_id, duration=value)
    await update.message.reply_text(f"Duracion del video: {value}s")


async def set_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    save_settings(update.effective_chat.id, system_prompt=text)
    await update.message.reply_text(f"System prompt actualizado:\n{text}" if text else "System prompt borrado.")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message

    tg_file = message.voice or message.audio
    if tg_file is None:
        await message.reply_text("Enviame un audio o una nota de voz.")
        return

    if _awaiting.get(chat_id) == "image":
        await message.reply_text("Estoy esperando una imagen, no un audio. Envia una foto (o /resetimage para cancelar).")
        _awaiting.pop(chat_id, None)
        return

    file = await context.bot.get_file(tg_file.file_id)
    ext = "ogg" if message.voice else (tg_file.file_name.split(".")[-1] if tg_file.file_name else "mp3")
    local_in_path = os.path.join(TMP_DIR, f"{chat_id}_{tg_file.file_unique_id}.{ext}")
    await file.download_to_drive(local_in_path)

    try:
        if _awaiting.get(chat_id) == "voice":
            _awaiting.pop(chat_id, None)
            await message.reply_text("Guardando voz, un momento...")
            ref_text = await asyncio.to_thread(save_voice_config, chat_id, local_in_path)
            await message.reply_text(f'Voz guardada. Transcripcion detectada: "{ref_text}"')
            return

        await message.reply_text("Transcribiendo...")
        prompt_text, _ = await asyncio.to_thread(transcribe, local_in_path)
        await message.reply_text(f'Dijiste: "{prompt_text}"')
        await respond(update, context, prompt_text)

    except Exception as e:
        log.exception("Error procesando audio")
        await message.reply_text(f"Ups, hubo un error: {e}")
    finally:
        if os.path.exists(local_in_path):
            os.remove(local_in_path)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message
    caption = (message.caption or "").strip().lower()

    if _awaiting.get(chat_id) != "image" and not caption.startswith("/setimage"):
        await message.reply_text("Si quieres usar esta imagen como personaje, usa /setimage antes (o envia la foto con /setimage como pie de foto).")
        return

    _awaiting.pop(chat_id, None)
    tg_file = message.photo[-1] if message.photo else message.document
    try:
        file = await context.bot.get_file(tg_file.file_id)
        data = bytes(await file.download_as_bytearray())
        if len(data) > MAX_IMAGE_BYTES:
            await message.reply_text("La imagen pesa demasiado (max 5 MB). Envia una mas liviana.")
            return
        with open(image_path(chat_id), "wb") as f:
            f.write(data)
        await message.reply_text("Imagen guardada. Se usara como 'imagen 1' en el modo video.")
    except Exception as e:
        log.exception("Error guardando imagen")
        await message.reply_text(f"Ups, hubo un error: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message
    text = message.text.strip()
    if not text:
        return

    if chat_id in _awaiting:
        kind = "un audio" if _awaiting[chat_id] == "voice" else "una imagen"
        await message.reply_text(f"Estoy esperando {kind}, no texto.")
        return

    try:
        await respond(update, context, text)
    except Exception as e:
        log.exception("Error procesando texto")
        await message.reply_text(f"Ups, hubo un error: {e}")


def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
        .get_updates_write_timeout(30)
        .get_updates_pool_timeout(30)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("mode", mode_cmd))
    app.add_handler(CommandHandler("setvoice", setvoice))
    app.add_handler(CommandHandler("resetvoice", resetvoice))
    app.add_handler(CommandHandler("setimage", setimage))
    app.add_handler(CommandHandler("resetimage", resetimage))
    app.add_handler(CommandHandler("duration", duration_cmd))
    app.add_handler(CommandHandler("system", set_system))
    app.add_handler(CallbackQueryHandler(mode_callback, pattern=r"^mode:"))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Bot corriendo...")
    app.run_polling()


if __name__ == "__main__":
    main()