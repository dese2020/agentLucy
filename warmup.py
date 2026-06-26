import os
import sys

# Verificar la version de vLLM y donde guarda el cache
def find_cache_vars():
    try:
        import vllm
        version = vllm.__version__
        print(f"vLLM version: {version}")
    except:
        print("No se pudo obtener version")

    # Buscar donde guarda el cache de compilacion
    try:
        from vllm.config import CompilationConfig
        import inspect
        src = inspect.getsource(CompilationConfig)
        # Buscar referencias a cache_dir
        for line in src.split('\n'):
            if 'cache' in line.lower() and ('dir' in line.lower() or 'path' in line.lower()):
                print("CompilationConfig:", line.strip())
    except Exception as e:
        print(f"CompilationConfig error: {e}")

    # Buscar variables de entorno relacionadas con cache
    try:
        from vllm import envs
        import inspect
        src = inspect.getsource(envs)
        for line in src.split('\n'):
            if 'cache' in line.lower() and ('VLLM' in line or 'compile' in line.lower()):
                print("envs:", line.strip())
    except Exception as e:
        print(f"envs error: {e}")

if __name__ == '__main__':
    find_cache_vars()

    from vllm import LLM, SamplingParams

    print("\n==> Cargando modelo y compilando Inductor (~2-3 min)...")
    llm = LLM(
        model="/models/qwen3.5-9b",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        kv_cache_dtype="fp8",
        enforce_eager=False,
        max_num_seqs=64,
    )

    print("==> Warmup con prompts...")
    params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate([
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum physics briefly.",
    ], params)

    for o in outputs:
        print("  ->", o.outputs[0].text[:80])

    # Mostrar donde quedo el cache
    print("\n==> Buscando cache generado...")
    import subprocess
    for cache_dir in ["/root/.cache/vllm", "/tmp/vllm", "/root/.vllm", "/vllm-cache"]:
        if os.path.exists(cache_dir):
            result = subprocess.run(["du", "-sh", cache_dir], capture_output=True, text=True)
            print(f"  {cache_dir}: {result.stdout.strip()}")
            subprocess.run(["find", cache_dir, "-name", "*.bin", "-o", "-name", "*.pt",
                          "-o", "-name", "*.pkl", "-o", "-name", "*.json"],
                         capture_output=False)

    print("\n==> Done.")
    del llm
