import base64
import json
import runpod

# 1. Configuración de credenciales y endpoint
RUNPOD_API_KEY = "TU_RUNPOD_API_KEY_AQUI"
ENDPOINT_ID = "TU_ENDPOINT_ID_AQUI"

runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(ENDPOINT_ID)


def convertir_workflow_a_api(archivo_json_path: str):
    """Lee un workflow UI visual de ComfyUI, lo codifica en Base64

    y lo envía al Handler de RunPod para convertirlo a formato API.
    """
    try:
        # 2. Leer el archivo JSON visual original
        with open(archivo_json_path, "r", encoding="utf-8") as f:
            contenido_json = f.read()

        # 3. Codificar el contenido JSON a Base64
        bytes_json = contenido_json.encode("utf-8")
        b64_str = base64.b64encode(bytes_json).decode("utf-8")

        print(f"📄 Enviando '{archivo_json_path}' codificado en Base64...")

        # 4. Enviar la solicitud síncrona al endpoint
        run_request = endpoint.run_sync(
            {
                "input": {
                    "debug": "convert_workflow",
                    "ui_workflow_b64": b64_str,
                }
            },
            timeout=60,  # Tiempo máximo de espera en segundos
        )

        # 5. Procesar la respuesta
        if "error" in run_request:
            print("❌ Error devuelto por el handler:", run_request["error"])
            return None

        api_workflow = run_request.get("api_workflow")
        print("✅ Conversion exitosa!")

        # Opcional: Guardar el resultado devuelto en un archivo local
        output_file = "workflow_api_convertido.json"
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(api_workflow, out, indent=2)
        print(f"💾 Resultado guardado en '{output_file}'")

        return api_workflow

    except Exception as e:
        print(f"💥 Ocurrió una excepción: {str(e)}")
        return None


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Cambia esto por la ruta a tu archivo JSON con el workflow de ComfyUI UI
    RUTA_MI_WORKFLOW = "ui_workflow_source.json"

    resultado = convertir_workflow_a_api(RUTA_MI_WORKFLOW)