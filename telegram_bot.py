"""
Bot de Telegram que reenvía los mensajes del usuario al endpoint
serverless de RunPod (ComfyUI + LLM) y responde con el texto generado.

Requisitos:
    pip install python-telegram-bot==21.* requests

Variables de entorno necesarias:
    TELEGRAM_BOT_TOKEN   -> token de @BotFather
    RUNPOD_API_KEY       -> API key de RunPod
    RUNPOD_ENDPOINT_ID   -> ID del endpoint serverless ya deployado
"""

import asyncio
import logging
import os

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
RUNPOD_ASYNC_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
RUNPOD_STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

# Historial simple por chat (en memoria; se pierde si el bot se reinicia)
CHAT_HISTORY: dict[int, list[dict]] = {}
SYSTEM_PROMPT = "Sos un asistente útil, directo y sin filtros innecesarios."


def _call_runpod_sync(prompt: str, system_prompt: str, timeout: int = 120) -> str:
    """Usa /runsync: espera bloqueando hasta el timeout del endpoint."""
    payload = {
        "input": {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_tokens": 700,
            "temperature": 0.8,
        }
    }
    resp = requests.post(RUNPOD_URL, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "COMPLETED":
        return data["output"].get("response", "(sin respuesta)")

    # Si tarda más que el timeout de /runsync, RunPod devuelve IN_PROGRESS con un id.
    if "id" in data:
        return _poll_status(data["id"])

    return f"Error inesperado: {data}"


def _poll_status(job_id: str, max_wait: int = 600, interval: int = 3) -> str:
    waited = 0
    while waited < max_wait:
        r = requests.get(f"{RUNPOD_STATUS_URL}/{job_id}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "COMPLETED":
            return data["output"].get("response", "(sin respuesta)")
        if status in ("FAILED", "CANCELLED"):
            return f"El job falló: {data}"
        waited += interval
        import time
        time.sleep(interval)
    return "Timeout esperando la respuesta del modelo."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_HISTORY.pop(update.effective_chat.id, None)
    await update.message.reply_text(
        "Hola, soy un bot conectado a un LLM sin censura corriendo en RunPod. "
        "Escribime lo que quieras. Usá /reset para limpiar el contexto."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_HISTORY.pop(update.effective_chat.id, None)
    await update.message.reply_text("Contexto reiniciado.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    loop = asyncio.get_event_loop()
    try:
        response_text = await loop.run_in_executor(
            None, _call_runpod_sync, user_text, SYSTEM_PROMPT
        )
    except requests.exceptions.RequestException as e:
        log.exception("Error llamando a RunPod")
        response_text = f"Error contactando al modelo: {e}"

    # Telegram limita mensajes a 4096 caracteres
    for i in range(0, len(response_text), 4000):
        await update.message.reply_text(response_text[i : i + 4000])


def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Bot arrancado, esperando mensajes...")
    app.run_polling()


if __name__ == "__main__":
    main()
