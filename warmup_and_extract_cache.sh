#!/bin/bash
set -e

ARCHIVE="/tmp/vllm-compile-cache.tar.gz"

echo "==> Ejecutando warmup (detecta version y cache automaticamente)..."
python3 /tmp/warmup.py 2>&1 | tee /tmp/warmup.log

echo ""
echo "==> Buscando directorios de cache para empaquetar..."
CACHE_FOUND=""

# Candidatos conocidos de cache de vLLM
for dir in "/root/.cache/vllm" "/root/.vllm" "/tmp/vllm" "/home/user/.cache/vllm"; do
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo "  Cache encontrado en: $dir ($(du -sh $dir | cut -f1))"
        CACHE_FOUND="$dir"
    fi
done

if [ -z "$CACHE_FOUND" ]; then
    echo "ERROR: No se encontro cache. Revisa /tmp/warmup.log"
    exit 1
fi

echo ""
echo "==> Comprimiendo $CACHE_FOUND ..."
tar -czf "$ARCHIVE" -C "$(dirname $CACHE_FOUND)" "$(basename $CACHE_FOUND)"
SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo "==> Listo: $ARCHIVE ($SIZE)"
echo ""
echo "Descarga con:"
echo "  scp -P <PUERTO> root@<IP>:$ARCHIVE ./"
