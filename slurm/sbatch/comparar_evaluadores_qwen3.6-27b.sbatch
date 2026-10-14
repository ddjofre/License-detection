#!/bin/bash
#SBATCH --job-name=cmp3
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# cmp3 = Qwen3.6-27B. Ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch para
# el mapeo completo de job-names cortos (cmp1=32B, cmp2=72B, cmp3=Qwen3.6-27B,
# cmp4=Qwen3-Next-80B-A3B).
#
# Corre 02_evaluar_licencias_llm.py con Qwen3.6-27B
# offline/batch (vLLM, sin servidor, 2 GPUs -- pesa 51.7 GiB bf16,
# confirmado via model.safetensors.index.json: entraría en 1 sola A100 80GB
# con ~27 GiB libres para KV cache, pero se usan 2 a pedido del usuario,
# mismo criterio de margen que ya usa el repo para Qwen2.5-32B -- 61 GiB en
# 2 GPUs en vez de ajustarlo a 1) sobre el mismo subconjunto fijo de 341
# URLs que slurm/comparar_evaluadores_qwen2.5-32b.sbatch.
#
# OJO: Qwen3.6-27B es multimodal (architectures: Qwen3_5ForConditionalGeneration,
# tiene vision_config e image_token_id en su config.json) -- confirmado
# 2026-10-01. Para este uso no importa: solo se le mandan prompts de texto
# (chat_template.jinja soporta texto solo), pero si en algún momento tira un
# error raro al cargar (ej. pidiendo un procesador de imágenes), ese es el
# motivo -- no es un problema del script.
#
# Ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch para el detalle completo
# de la comparación (subconjunto, carpeta de salida por modelo, tiempos.csv
# compartido, motivo de la cuenta latamgpt/partition=lascar/prefijo -ranokau).
#
# Salida: salida/qwen3.6-27b/licencias_evaluadas_qwen3.6-27b.csv
# (slug del modelo en la carpeta Y en el nombre del archivo, a propósito --
# ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch)

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export CMP_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen3.6-27B
export CMP_ENTRADA="salida/comparacion_modelos/subset_urls.jsonl"
export CMP_TENSOR_PARALLEL_SIZE=2
export CMP_GPU_MEMORY_UTILIZATION=0.9
export CMP_MAX_MODEL_LEN=4096
export CMP_TIEMPOS="salida/comparacion_modelos/tiempos.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
