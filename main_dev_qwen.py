from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from openai import AsyncOpenAI
import asyncio, os, re, logging, io, base64, random
import httpx
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()

BOT_TOKEN = os.getenv("api_token_telegram")
RUNPOD_API_KEY = os.getenv("api_token_runpod")
RUNPOD_ENDPOINT_ID = os.getenv("runpod_endpoint_id", "xuqx9qnhz7un1m")
RUNPOD_REST_URL = f"https://rest.runpod.io/v1/endpoints/{RUNPOD_ENDPOINT_ID}"

# Endpoint del serverless de generación de imágenes (Qwen Image Edit)
IMAGE_GEN_ENDPOINT_ID = os.getenv("runpod_image_endpoint_id", "avvuq4xc0u5boa")
IMAGE_GEN_RUN_URL = f"https://api.runpod.ai/v2/{IMAGE_GEN_ENDPOINT_ID}/run"
IMAGE_GEN_STATUS_URL = f"https://api.runpod.ai/v2/{IMAGE_GEN_ENDPOINT_ID}/status"

# Timeouts para el polling de generación de imagen
IMAGE_POLL_INTERVAL = 3      # segundos entre cada consulta de status
IMAGE_POLL_TIMEOUT = 180     # segundos máximos de espera

client = AsyncOpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
)

histories: dict[int, list] = {}
history_locks: dict[int, asyncio.Lock] = {}

active_agent_users: set[int] = set()
active_agent_lock = asyncio.Lock()

RUNPOD_CONCURRENCY = 10
runpod_semaphore = asyncio.Semaphore(RUNPOD_CONCURRENCY)

user_semaphores: dict[int, asyncio.Semaphore] = {}

MAX_CONTEXT_CHARS = (8192 - 1024) * 3

MODEL = "DavidAU/Qwen3.5-4B-Deckard-HERETIC-UNCENSORED-Thinking"
#MODEL = "ndimensional/qwen3.5-9b-uncensored-safetensors"
#MODEL = "nvidia/qwen3.6-35b-a3b-nvfp4"

# Almacena el último prompt generado por usuario para usarlo al pulsar el botón
# { user_id: "prompt text" }
last_prompts: dict[int, str] = {}

# Almacena el base64 de la última imagen enviada por el usuario
# { user_id: "base64string" }
last_images: dict[int, str] = {}

# Dimensiones de salida para el serverless de imagen
IMAGE_WIDTH = int(os.getenv("image_width", "480"))
IMAGE_HEIGHT = int(os.getenv("image_height", "832"))


# ---------------------------------------------------------------------------
# Helpers de contexto
# ---------------------------------------------------------------------------

def estimate_chars(messages: list) -> int:
    total = 0
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    total += len(block.get("text", ""))
    return total


def trim_history(messages: list) -> list:
    if estimate_chars(messages) <= MAX_CONTEXT_CHARS:
        return messages
    system = messages[:1]
    rest = messages[1:]
    while len(rest) > 1 and estimate_chars(system + rest) > MAX_CONTEXT_CHARS:
        rest = rest[1:]
    return system + rest


def get_user_semaphore(user_id: int) -> asyncio.Semaphore:
    if user_id not in user_semaphores:
        user_semaphores[user_id] = asyncio.Semaphore(1)
    return user_semaphores[user_id]


def get_history_lock(user_id: int) -> asyncio.Lock:
    if user_id not in history_locks:
        history_locks[user_id] = asyncio.Lock()
    return history_locks[user_id]


def init_history(user_id: int) -> None:
    systemConf = (
        "Rol: Eres Lucy una mujer coqueta, seductora y caliente. tu objetivo es complacer al usuario. usa emoticones de amor y sensuales."
    )
    histories[user_id] = [
        {"role": "system", "content": systemConf}
    ]


def extract_answer(raw: str, user_id: int) -> str:
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        logging.info(
            "[user=%s] <think>\n%s\n</think>",
            user_id, think_match.group(1).strip()
        )
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def extract_prompt_from_answer(answer: str) -> str | None:
    """
    Extrae un prompt de imagen desde la respuesta del agente.
    Prioridad:
    1. Bloque entre qw_edit ... fin_edit
    2. Texto entre comillas dobles
    3. Bloque en backticks
    4. Heurística de texto completo
    """

    # 🔥 Prioridad 1: qw_edit ... fin_edit
    match = re.search(r"qw_edit\s*(.*?)\s*fin_edit", answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Prioridad 2: texto entre comillas dobles
    match = re.search(r'"([^"]{20,})"', answer)
    if match:
        return match.group(1).strip()

    # Prioridad 3: bloque de código inline con backticks
    match = re.search(r'`([^`]{20,})`', answer)
    if match:
        return match.group(1).strip()

    # Prioridad 4: heurística (respuesta completa como prompt)
    cleaned = answer.strip()
    if len(cleaned) >= 30 and "?" not in cleaned and cleaned[0].isupper():
        return cleaned

    return None


async def build_image_content(photo, context) -> tuple[list, str]:
    """Descarga la foto y devuelve (content_block, base64_string)."""
    file = await context.bot.get_file(photo[-1].file_id)
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        }
    ]
    return content, b64


# ---------------------------------------------------------------------------
# Generación de imagen con RunPod (asíncrono + polling)
# ---------------------------------------------------------------------------

async def submit_image_job(prompt: str, image_b64: str) -> str:
    """Envía el trabajo al serverless y devuelve el job_id."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "image_base64": image_b64,
            "seed": random.randint(0, 2**32 - 1),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        }
    }
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.post(IMAGE_GEN_RUN_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["id"]


async def poll_image_job(job_id: str) -> dict:
    """
    Hace polling hasta que el job termine.
    Devuelve el dict de output del job o lanza TimeoutError.
    """
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    deadline = asyncio.get_event_loop().time() + IMAGE_POLL_TIMEOUT

    async with httpx.AsyncClient(timeout=30) as http:
        while True:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Job {job_id} no terminó en {IMAGE_POLL_TIMEOUT}s")

            resp = await http.get(
                f"{IMAGE_GEN_STATUS_URL}/{job_id}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "COMPLETED":
                return data.get("output", {})
            elif status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Job {job_id} terminó con status: {status}")

            await asyncio.sleep(IMAGE_POLL_INTERVAL)


def decode_image_output(output: dict) -> bytes | None:
    """
    Intenta extraer bytes de imagen del output.
    Soporta: { "image": "<base64>" } o { "images": ["<base64>", ...] }
    o una URL directa en { "image_url": "..." }.
    """
    # Caso base64 directo
    for key in ("image", "images"):
        val = output.get(key)
        if val:
            b64_str = val[0] if isinstance(val, list) else val
            # Limpiar prefijos data URI si los hay
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            return base64.b64decode(b64_str)

    return None


async def generate_image(prompt: str, image_b64: str) -> bytes:
    """Pipeline completo: submit → poll → decode. Devuelve bytes de la imagen."""
    job_id = await submit_image_job(prompt, image_b64)
    logging.info("Image job submitted: %s", job_id)
    output = await poll_image_job(job_id)
    image_bytes = decode_image_output(output)
    if image_bytes is None:
        raise ValueError(f"No se encontró imagen en el output del job: {output}")
    return image_bytes


# ---------------------------------------------------------------------------
# Handlers de Telegram
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    async with get_history_lock(user_id):
        init_history(user_id)
    await update.message.reply_text("Hola, soy Lucy. Envíame una imagen o escríbeme algo. 😉")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    async with get_history_lock(user_id):
        init_history(user_id)
    last_prompts.pop(user_id, None)
    last_images.pop(user_id, None)
    await update.message.reply_text("Historial borrado. Empezamos de cero.")


async def _send_agent_reply(
    update: Update,
    waiting_msg,
    answer: str,
    user_id: int,
):
    """
    Edita el mensaje de espera con la respuesta del agente.
    Si detecta un prompt de imagen, añade el botón inline de generación.
    """
    #prompt = extract_prompt_from_answer(answer)
    prompt = False
    print(f"ANSWER: {answer}")

    if prompt:
        last_prompts[user_id] = prompt
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🎨 Generar imagen", callback_data=f"gen_img:{user_id}")]
        ])
        await waiting_msg.edit_text(answer, reply_markup=keyboard)
    else:
        last_prompts.pop(user_id, None)
        await waiting_msg.edit_text(answer)


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = update.message.text
    print(f"USER: {prompt}")

    user_sem = get_user_semaphore(user_id)
    if user_sem.locked():
        await update.message.reply_text(
            "Aún estoy procesando tu mensaje anterior, espera un momento."
        )
        return

    async with user_sem:
        lock = get_history_lock(user_id)

        async with lock:
            if user_id not in histories:
                init_history(user_id)
            histories[user_id].append({"role": "user", "content": prompt})
            messages_snapshot = trim_history(list(histories[user_id]))

        waiting = await update.message.reply_text("Pensando...")

        try:
            async with runpod_semaphore:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages_snapshot,
                    temperature=0.7,
                    max_tokens=1024,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                )

            answer = extract_answer(response.choices[0].message.content, user_id)

            async with lock:
                histories[user_id].append({"role": "assistant", "content": answer})
                if len(histories[user_id]) > 30:
                    histories[user_id] = (
                        histories[user_id][:1] + histories[user_id][-29:]
                    )

            await _send_agent_reply(update, waiting, answer, user_id)

        except Exception as e:
            logging.exception("[user=%s] Error al llamar al modelo: %s", user_id, e)
            async with lock:
                if histories.get(user_id):
                    histories[user_id].pop()
            await waiting.edit_text(f"Error al contactar el modelo: {e}")


async def chat_with_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    caption = update.message.caption or "Describe esta imagen."

    user_sem = get_user_semaphore(user_id)
    if user_sem.locked():
        await update.message.reply_text(
            "Aún estoy procesando tu mensaje anterior, espera un momento."
        )
        return

    async with user_sem:
        lock = get_history_lock(user_id)

        image_content, image_b64 = await build_image_content(update.message.photo, context)
        last_images[user_id] = image_b64

        user_message = {
            "role": "user",
            "content": image_content + [{"type": "text", "text": caption}]
        }

        async with lock:
            if user_id not in histories:
                init_history(user_id)
            histories[user_id].append(user_message)
            messages_snapshot = trim_history(list(histories[user_id]))

        waiting = await update.message.reply_text("Analizando imagen...")

        try:
            async with runpod_semaphore:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages_snapshot,
                    temperature=0.7,
                    max_tokens=1024,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                )

            answer = extract_answer(response.choices[0].message.content, user_id)

            async with lock:
                histories[user_id].append({"role": "assistant", "content": answer})
                if len(histories[user_id]) > 30:
                    histories[user_id] = (
                        histories[user_id][:1] + histories[user_id][-29:]
                    )

            await _send_agent_reply(update, waiting, answer, user_id)

        except Exception as e:
            logging.exception("[user=%s] Error procesando imagen: %s", user_id, e)
            async with lock:
                if histories.get(user_id):
                    histories[user_id].pop()
            await waiting.edit_text(f"Error al procesar la imagen: {e}")


async def generate_image_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Manejador del botón inline '🎨 Generar imagen'.
    Recupera el último prompt guardado para ese usuario y llama al serverless.
    """
    query = update.callback_query
    await query.answer()  # quita el spinner del botón

    # El callback_data tiene formato "gen_img:<user_id>"
    parts = query.data.split(":", 1)
    if len(parts) != 2:
        return
    owner_id = int(parts[1])

    # Solo el dueño del prompt puede generarlo
    requester_id = query.from_user.id
    if requester_id != owner_id:
        await query.answer("Este botón es solo para el usuario que pidió el prompt.", show_alert=True)
        return

    prompt = last_prompts.get(owner_id)
    image_b64 = last_images.get(owner_id)

    if not prompt:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("⚠️ Ya no tengo el prompt guardado. Pide uno nuevo.")
        return

    if not image_b64:
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text("⚠️ Ya no tengo la imagen original guardada. Envíala de nuevo junto con tu petición.")
        return

    # Deshabilitar el botón mientras se genera
    await query.edit_message_reply_markup(reply_markup=None)
    status_msg = await query.message.reply_text("🎨 Generando imagen, esto puede tardar un poco...")

    try:
        image_bytes = await generate_image(prompt, image_b64)
        buf = io.BytesIO(image_bytes)
        buf.name = "generated.jpg"
        await status_msg.delete()
        await query.message.reply_photo(
            photo=buf,
            caption=f"✨ Prompt usado:\n`{prompt}`",
            parse_mode="Markdown",
        )
        logging.info("[user=%s] Imagen generada con prompt: %s", owner_id, prompt[:80])

    except TimeoutError:
        await status_msg.edit_text("⏱️ La generación tardó demasiado. Intenta de nuevo.")
        logging.warning("[user=%s] Timeout esperando imagen", owner_id)
    except Exception as e:
        await status_msg.edit_text(f"❌ Error al generar la imagen: {e}")
        logging.exception("[user=%s] Error generando imagen: %s", owner_id, e)


# ---------------------------------------------------------------------------
# RunPod worker management (sin cambios)
# ---------------------------------------------------------------------------

async def set_workers_min(workers_min: int) -> dict:
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"workersMin": workers_min}
    async with httpx.AsyncClient(timeout=15) as http:
        response = await http.patch(RUNPOD_REST_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


async def activar_agente(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = await update.message.reply_text("⏳ Activando agente...")

    async with active_agent_lock:
        ya_activo = user_id in active_agent_users
        active_agent_users.add(user_id)
        total = len(active_agent_users)

        if ya_activo:
            await msg.edit_text(
                f"✅ Ya tenías el agente activado.\n"
                f"• Usuarios con agente activo: {total}"
            )
            return

        should_call_runpod = total == 1

    if should_call_runpod:
        try:
            data = await set_workers_min(1)
            workers_min = data.get("workersMin", "?")
            workers_max = data.get("workersMax", "?")
            await msg.edit_text(
                f"✅ Agente activado.\n"
                f"• Workers mínimos: {workers_min} | máximos: {workers_max}\n"
                f"• Usuarios con agente activo: {total}\n\n"
                f"Worker precalentado y listo. 🔥"
            )
            logging.info("RunPod workersMin=1 activado por user=%s (total activos: %s)", user_id, total)
        except httpx.HTTPStatusError as e:
            async with active_agent_lock:
                active_agent_users.discard(user_id)
            logging.error("Error HTTP al activar agente: %s", e)
            await msg.edit_text(f"❌ Error al activar en RunPod (HTTP {e.response.status_code}):\n{e.response.text}")
        except Exception as e:
            async with active_agent_lock:
                active_agent_users.discard(user_id)
            logging.exception("Error inesperado al activar agente: %s", e)
            await msg.edit_text(f"❌ Error inesperado: {e}")
    else:
        await msg.edit_text(
            f"✅ Te uniste al agente (ya estaba activo por otro usuario).\n"
            f"• Usuarios con agente activo: {total}"
        )
        logging.info("user=%s se unió al agente (ya activo, total: %s)", user_id, total)


async def desactivar_agente(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg = await update.message.reply_text("⏳ Desactivando agente...")

    async with active_agent_lock:
        if user_id not in active_agent_users:
            await msg.edit_text("ℹ️ No tenías el agente activado.")
            return

        active_agent_users.discard(user_id)
        restantes = len(active_agent_users)
        should_call_runpod = restantes == 0

    if should_call_runpod:
        try:
            data = await set_workers_min(0)
            workers_min = data.get("workersMin", "?")
            await msg.edit_text(
                f"🛑 Agente desactivado.\n"
                f"• Workers mínimos: {workers_min}\n"
                f"• Usuarios activos restantes: 0\n\n"
                f"Ningún usuario activo — RunPod ya no mantiene workers. 💤"
            )
            logging.info("RunPod workersMin=0 desactivado por user=%s (sin activos)", user_id)
        except httpx.HTTPStatusError as e:
            async with active_agent_lock:
                active_agent_users.add(user_id)
            logging.error("Error HTTP al desactivar agente: %s", e)
            await msg.edit_text(f"❌ Error al desactivar en RunPod (HTTP {e.response.status_code}):\n{e.response.text}")
        except Exception as e:
            async with active_agent_lock:
                active_agent_users.add(user_id)
            logging.exception("Error inesperado al desactivar agente: %s", e)
            await msg.edit_text(f"❌ Error inesperado: {e}")
    else:
        await msg.edit_text(
            f"✅ Tu agente fue desactivado.\n"
            f"• Usuarios con agente activo restantes: {restantes}\n\n"
            f"El worker sigue activo por los demás usuarios. 🔥"
        )
        logging.info("user=%s desactivó su agente (restantes activos: %s)", user_id, restantes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("activarAgente", activar_agente))
    app.add_handler(CommandHandler("desactivarAgente", desactivar_agente))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )
    app.add_handler(
        MessageHandler(filters.PHOTO, chat_with_photo)
    )
    # Handler del botón inline de generación de imagen
    app.add_handler(
        CallbackQueryHandler(generate_image_callback, pattern=r"^gen_img:\d+$")
    )

    app.run_polling()


if __name__ == "__main__":
    main()
