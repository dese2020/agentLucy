"""
Bot de Telegram que habla con el endpoint serverless de RunPod
(ComfyUI + Qwen3.5 Heretic Uncensored LLM + Fish Audio S2-Pro TTS/voice cloning).
"""

import asyncio
import base64
import logging
import os
import time
from dotenv import load_dotenv
import requests
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

CHAT_HISTORY: dict[int, list[dict]] = {}
PENDING_VOICE_REF: dict[int, dict] = {}
SYSTEM_PROMPT = "Sos un asistente útil, directo y sin filtros innecesarios."


def _call_runpod(payload: dict, timeout: int = 180) -> dict:
    resp = requests.post(RUNPOD_URL, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "COMPLETED":
        return data.get("output", {})

    if "id" in data:
        return _poll_status(data["id"])

    return {"error": f"Error inesperado: {data}"}


def _poll_status(job_id: str, max_wait: int = 600, interval: int = 3) -> dict:
    waited = 0
    while waited < max_wait:
        r = requests.get(f"{RUNPOD_STATUS_URL}/{job_id}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")

        if status == "COMPLETED":
            return data.get("output", {})
        if status in ("FAILED", "CANCELLED"):
            return {"error": f"El job falló: {data}"}

        waited += interval
        time.sleep(interval)

    return {"error": "Timeout esperando la respuesta del modelo."}


def _call_runpod_llm(prompt: str, system_prompt: str) -> str:
    payload = {
        "input": {
            "mode": "llm",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_length": 1024,
            "sampling_mode": "on",
            "thinking": True,
            "use_default_template": True,
            "temperature": 0.8,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.05,
            "repetition_penalty": 1.05,
            "presence_penalty": 0.0,
            "seed": 0,
        }
    }
    output = _call_runpod(payload)
    if "error" in output:
        return output["error"]
    return output.get("response", "(sin respuesta)")


def _call_runpod_tts(text: str) -> dict:
    payload = {"input": {"mode": "tts", "text": text, "language": "auto"}}
    return _call_runpod(payload)


def _call_runpod_voice_clone(text: str, reference_audio_b64: str, audio_format: str, reference_text: str = "") -> dict:
    payload = {
        "input": {
            "mode": "voice_clone",
            "text": text,
            "reference_audio_b64": reference_audio_b64,
            "reference_audio_format": audio_format,
            "reference_text": reference_text,  # <-- Se envía a RunPod
        }
    }
    return _call_runpod(payload, timeout=240)


async def _send_audio_output(update: Update, output: dict, caption: str = None):
    if "error" in output:
        await update.message.reply_text(f"Error generando audio: {output['error']}")
        return

    audio_b64 = output.get("audio_base64")
    if not audio_b64:
        await update.message.reply_text(f"Respuesta inesperada del modelo: {output}")
        return

    audio_bytes = base64.b64decode(audio_b64)
    audio_format = output.get("audio_format", "mp3")
    filename = f"output.{audio_format}"

    await update.message.reply_audio(
        audio=audio_bytes,
        filename=filename,
        caption=caption,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_HISTORY.pop(update.effective_chat.id, None)
    PENDING_VOICE_REF.pop(update.effective_chat.id, None)
    await update.message.reply_text(
        "Hola! Soy un bot conectado a Qwen3.5 Heretic y a Fish Audio (TTS) corriendo en RunPod.\n\n"
        "- Escribime lo que quieras para chatear con el LLM.\n"
        "- /tts <texto> para generar audio con la voz por defecto.\n"
        "- Mandame una nota de voz para usarla como referencia, y después "
        "/clonar <texto> para generar audio con esa voz.\n"
        "- /voz_default para olvidar la voz de referencia guardada.\n"
        "- /reset para limpiar el contexto del chat."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_HISTORY.pop(update.effective_chat.id, None)
    await update.message.reply_text("Contexto reiniciado.")


async def voz_default(update: Update, context: ContextTypes.DEFAULT_TYPE):
    PENDING_VOICE_REF.pop(update.effective_chat.id, None)
    await update.message.reply_text("Listo, olvidé la voz de referencia guardada.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    loop = asyncio.get_running_loop()
    try:
        response_text = await loop.run_in_executor(
            None, _call_runpod_llm, user_text, SYSTEM_PROMPT
        )
    except requests.exceptions.RequestException as e:
        log.exception("Error llamando a RunPod")
        response_text = f"Error contactando al modelo: {e}"

    for i in range(0, len(response_text), 4000):
        await update.message.reply_text(response_text[i : i + 4000])


async def handle_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("Uso: /tts <texto a convertir en audio>")
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE
    )

    loop = asyncio.get_running_loop()
    try:
        output = await loop.run_in_executor(None, _call_runpod_tts, text)
    except requests.exceptions.RequestException as e:
        log.exception("Error llamando a RunPod (tts)")
        await update.message.reply_text(f"Error contactando al modelo: {e}")
        return

    await _send_audio_output(update, output)


async def handle_clonar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("Uso: /clonar <texto a decir con la voz guardada>")
        return

    ref = PENDING_VOICE_REF.get(chat_id)
    if not ref:
        await update.message.reply_text(
            "Todavía no me mandaste ninguna nota de voz para clonar. "
            "Mandame un audio primero y después usá /clonar <texto>."
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)

    loop = asyncio.get_running_loop()
    try:
        output = await loop.run_in_executor(
            None,
            _call_runpod_voice_clone,
            text,
            ref["audio_b64"],
            ref["format"],
            ref.get("ref_text", ""),
        )
    except requests.exceptions.RequestException as e:
        log.exception("Error llamando a RunPod (voice_clone)")
        await update.message.reply_text(f"Error contactando al modelo: {e}")
        return

    await _send_audio_output(update, output)


async def handle_voice_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    voice = update.message.voice or update.message.audio
    if voice is None:
        return

    # Si el usuario escribió un texto junto al audio/nota de voz (caption)
    ref_text = update.message.caption or ""

    tg_file = await context.bot.get_file(voice.file_id)
    audio_bytes = await tg_file.download_as_bytearray()
    audio_b64 = base64.b64encode(bytes(audio_bytes)).decode("utf-8")

    audio_format = "ogg" if update.message.voice else (voice.mime_type or "").split("/")[-1] or "ogg"

    PENDING_VOICE_REF[chat_id] = {
        "audio_b64": audio_b64,
        "format": audio_format,
        "ref_text": ref_text,
    }

    msg = "Guardé esa voz como referencia."
    if ref_text:
        msg += f'\n📝 Transcripción guardada: "{ref_text}"'
    else:
        msg += "\n💡 *Tip*: Si le agregás un texto (caption) a la nota de voz con lo que dijiste, la clonación será mucho más precisa."

    await update.message.reply_text(msg, parse_mode="Markdown")


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("tts", handle_tts))
    app.add_handler(CommandHandler("clonar", handle_clonar))
    app.add_handler(CommandHandler("voz_default", voz_default))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_note))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot arrancado, esperando mensajes...")
    app.run_polling()


if __name__ == "__main__":
    main()