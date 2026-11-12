"""
Bot de Telegram para tu endpoint serverless de RunPod (ComfyUI: Qwen + Whisper + FishAudio).

Flujo:
  1. El usuario manda un audio/voice por Telegram -> se transcribe con Whisper (debug="transcribe")
  2. El texto transcrito se manda a Qwen (mode="llm")
  3. La respuesta se convierte a audio:
       - mode="tts"          -> voz por defecto de FishAudio
       - mode="voice_clone"  -> si el usuario configuró una voz de referencia con /setvoice

La voz clonada se guarda POR USUARIO (cada chat_id tiene su propia configuración),
en un archivo voices/<chat_id>.json.

Requisitos:
    pip install python-telegram-bot==21.* requests

Variables de entorno:
    export TELEGRAM_BOT_TOKEN="token_de_botfather"
    export RUNPOD_API_KEY="tu_api_key"
    export RUNPOD_ENDPOINT_ID="tu_endpoint_id"

Uso:
    python qwen_telegram_bot.py

Comandos en Telegram:
    /start            - instrucciones
    /setvoice         - respondé a este mensaje reenviando un audio de referencia,
                        o simplemente mandá un audio después de este comando
    /resetvoice       - vuelve a la voz por defecto
    /system <texto>   - fija un system prompt para tus próximas consultas
    (mandar un audio/voice cualquiera) - hace la pregunta al modelo
"""

import base64
import json
import logging
import os
import re
import time
from dotenv import load_dotenv
import requests
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
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

for name, val in [
    ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
    ("RUNPOD_API_KEY", RUNPOD_API_KEY),
    ("RUNPOD_ENDPOINT_ID", RUNPOD_ENDPOINT_ID),
]:
    if not val:
        raise SystemExit(f"Falta la variable de entorno {name}")

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# chat_id -> "esperando audio para /setvoice"
_awaiting_voice_setup = set()
# chat_id -> system prompt en memoria
_system_prompts = {}


# ---------------------------------------------------------------------------
# Llamadas al endpoint serverless
# ---------------------------------------------------------------------------
def run_job(payload, timeout=600):
    resp = requests.post(f"{BASE_URL}/runsync", headers=HEADERS, json={"input": payload}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        job_id = data["id"]
        start = time.time()
        while time.time() - start < timeout:
            r = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "COMPLETED":
                break
            if data.get("status") == "FAILED":
                raise RuntimeError(f"Job falló: {data}")
            time.sleep(2)

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


def transcribe(audio_path, language="auto"):
    ext = os.path.splitext(audio_path)[1].lstrip(".") or "ogg"
    payload = {
        "debug": "transcribe",
        "reference_audio_b64": file_to_b64(audio_path),
        "reference_audio_format": ext,
        "language": language,
    }
    out = run_job(payload)
    return out["transcription"], out.get("detected_language")


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Fish Audio S2-Pro corta en ~1024 tokens de audio -> ~47 segundos.
# A una tasa de habla normal (~2.3 palabras/seg en español) eso equivale
# a más o menos 100 palabras / 620 caracteres. Dejamos margen de seguridad.
TTS_MAX_WORDS = 350
TTS_MAX_CHARS = 600
BREVITY_HINT = (
    "Responde siempre de forma breve y directa, en como máximo "
    f"{TTS_MAX_WORDS} palabras, porque tu respuesta se va a convertir a audio "
    "y el audio tiene un límite de duración."
)


def strip_thinking(text):
    """Elimina el bloque <think>...</think> que Qwen puede incluir en su salida."""
    cleaned = THINK_TAG_RE.sub("", text)
    return cleaned.strip()


def truncate_for_tts(text, max_chars=TTS_MAX_CHARS, max_words=TTS_MAX_WORDS):
    """
    Corta el texto para que nunca exceda lo que el TTS puede generar
    (max_new_tokens=1024 en el handler, ~47s de audio).
    Corta en el límite de oración/palabra más cercano para no dejar
    la frase picada a la mitad.
    """
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if len(text) > max_chars:
        text = text[:max_chars]

    # Intentamos cortar en el último punto/signo de cierre para que quede prolijo
    for sep in (". ", "! ", "? ", "\n"):
        idx = text.rfind(sep)
        if idx > max_chars * 0.4:  # no cortar demasiado corto
            text = text[: idx + 1]
            break
    else:
        # si no hay un buen punto de corte, al menos no cortamos a mitad de palabra
        text = text.rsplit(" ", 1)[0]

    return text.strip()


def ask_llm(prompt, system_prompt=None):
    # Combinamos el aviso de brevedad con el system prompt que haya puesto el usuario,
    # para que Qwen intente por su cuenta no pasarse del límite de audio.
    combined_system = f"{system_prompt}\n\n{BREVITY_HINT}" if system_prompt else BREVITY_HINT
    # "thinking": False le pide al nodo TextGenerate que no razone paso a paso.
    # Si el handler ignora esta clave, la limpieza de <think> abajo actúa como respaldo.
    payload = {"mode": "llm", "prompt": prompt, "thinking": False, "system_prompt": combined_system}
    out = run_job(payload)
    return strip_thinking(out["response"])


def text_to_speech(text, language="auto"):
    payload = {"mode": "tts", "text": text, "language": language}
    out = run_job(payload)
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
    out = run_job(payload)
    return out["audio_base64"], out["audio_format"]


# ---------------------------------------------------------------------------
# Config de voz por usuario (chat_id)
# ---------------------------------------------------------------------------
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
    path = voice_path(chat_id)
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Handlers de Telegram
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola! Mandame un audio y te respondo con otro audio.\n\n"
        "Comandos:\n"
        "/setvoice - después de este comando, mandá un audio de referencia para clonar esa voz\n"
        "/resetvoice - volvés a la voz por defecto\n"
        "/system <texto> - fija un system prompt para tus próximas preguntas"
    )


async def setvoice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    _awaiting_voice_setup.add(update.effective_chat.id)
    await update.message.reply_text("Ok, ahora mandame el audio que quiero usar como referencia de voz.")


async def resetvoice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_voice_config(update.effective_chat.id)
    await update.message.reply_text("Listo, vuelvo a la voz por defecto.")


async def set_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    chat_id = update.effective_chat.id
    if not text:
        _system_prompts.pop(chat_id, None)
        await update.message.reply_text("System prompt borrado.")
        return
    _system_prompts[chat_id] = text
    await update.message.reply_text(f"System prompt actualizado:\n{text}")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message

    # Telegram puede mandar 'voice' (nota de voz) o 'audio' (archivo de audio)
    tg_file = message.voice or message.audio
    if tg_file is None:
        await message.reply_text("Mandame un audio o una nota de voz.")
        return

    file = await context.bot.get_file(tg_file.file_id)
    ext = "ogg" if message.voice else (tg_file.file_name.split(".")[-1] if tg_file.file_name else "mp3")
    local_in_path = os.path.join(TMP_DIR, f"{chat_id}_{tg_file.file_unique_id}.{ext}")
    await file.download_to_drive(local_in_path)

    try:
        # --- Modo configuración de voz ---
        if chat_id in _awaiting_voice_setup:
            _awaiting_voice_setup.discard(chat_id)
            await message.reply_text("Clonando voz, un momento...")
            ref_text = save_voice_config(chat_id, local_in_path)
            await message.reply_text(f'Voz guardada. Transcripción detectada: "{ref_text}"')
            return

        # --- Modo conversación normal ---
        await message.reply_text("Transcribiendo...")
        prompt_text, _ = transcribe(local_in_path)
        await message.reply_text(f'Vos dijiste: "{prompt_text}"')

        await reply_with_audio(update, context, prompt_text)

    except Exception as e:
        log.exception("Error procesando audio")
        await message.reply_text(f"Ups, hubo un error: {e}")
    finally:
        if os.path.exists(local_in_path):
            os.remove(local_in_path)


async def reply_with_audio(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_text: str):
    """Le pregunta al LLM y responde con audio (usado por texto y por audio)."""
    chat_id = update.effective_chat.id
    message = update.message

    await message.reply_text("Pensando...")
    system_prompt = _system_prompts.get(chat_id)
    response_text = ask_llm(prompt_text, system_prompt=system_prompt)

    # Respaldo por si Qwen no respetó el pedido de brevedad: nunca mandamos
    # a TTS más texto del que entra en el límite de audio del handler.
    tts_text = truncate_for_tts(response_text)

    await message.reply_text(response_text if tts_text == response_text else f"{tts_text}\n\n(respuesta recortada para el audio)")

    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
    await message.reply_text("Generando audio...")

    voice_cfg = load_voice_config(chat_id)
    if voice_cfg:
        audio_b64, audio_fmt = text_to_speech_cloned(tts_text, voice_cfg)
    else:
        audio_b64, audio_fmt = text_to_speech(tts_text)

    out_path = os.path.join(TMP_DIR, f"{chat_id}_out.{audio_fmt}")
    b64_to_file(audio_b64, out_path)

    with open(out_path, "rb") as f:
        if audio_fmt in ("ogg", "oga"):
            await context.bot.send_voice(chat_id=chat_id, voice=f)
        else:
            await context.bot.send_audio(chat_id=chat_id, audio=f)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    message = update.message
    text = message.text.strip()

    if not text:
        return

    # Si el usuario había pedido /setvoice, un texto no sirve de referencia
    if chat_id in _awaiting_voice_setup:
        await message.reply_text("Necesito que mandes un audio, no texto, para clonar la voz.")
        return

    try:
        await reply_with_audio(update, context, text)
    except Exception as e:
        log.exception("Error procesando texto")
        await message.reply_text(f"Ups, hubo un error: {e}")


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setvoice", setvoice))
    app.add_handler(CommandHandler("resetvoice", resetvoice))
    app.add_handler(CommandHandler("system", set_system))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    log.info("Bot corriendo...")
    app.run_polling()


if __name__ == "__main__":
    main()
