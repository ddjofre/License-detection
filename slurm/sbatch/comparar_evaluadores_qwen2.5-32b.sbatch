#!/bin/bash
#SBATCH --job-name=cmp1
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# cmp1 = Qwen2.5-32B-Instruct. Job-name corto a propósito (no lleva el nombre
# del modelo) -- ver el resto de slurm/comparar_evaluadores_*.sbatch para el
# mapeo completo (cmp1=32B, cmp2=72B, cmp3=Qwen3.6-27B, cmp4=Qwen3-Next-80B-A3B).
#
# Corre 02_evaluar_licencias_llm.py con Qwen2.5-32B-Instruct
# offline/batch (vLLM, sin servidor, 2 GPUs -- mismo peso que ya usa
# vllm_server_licencias.sbatch: 61.0 GiB bf16, confirmado via
# model.safetensors.index.json) sobre el subconjunto fijo de 341 URLs
# (salida/comparacion_modelos/subset_urls.jsonl, sin las
# URLs "Alianza" de referencia/licencias_original_a_mano.xlsx -- no tienen licencia
# real que comparar).
#
# Parte de la comparación de tiempos/resultados entre versiones de Qwen --
# ver slurm/comparar_evaluadores_qwen2.5-72b.sbatch,
# slurm/comparar_evaluadores_qwen3.6-27b.sbatch y
# slurm/comparar_evaluadores_qwen3-next-80b-a3b.sbatch para los otros
# modelos de la misma comparación, todos sobre el mismo subconjunto.
#
# Salida: salida/qwen2.5-32b-instruct/licencias_evaluadas_qwen2.5-32b-instruct.csv
# (slug del modelo en la carpeta Y en el nombre del archivo, a propósito --
# así el csv se sigue identificando solo si se copia fuera de su carpeta)
# (una carpeta por modelo, mismo patrón que detectar_licencias_evaluar_qwen32b.sbatch)
# + una fila agregada en salida/comparacion_modelos/tiempos.csv
# (compartido entre todas las corridas de esta comparación).
#
# Mismo motivo que vllm_server_licencias.sbatch para la cuenta latamgpt +
# partition=lascar (único combo con GPU para esa cuenta) y el prefijo
# /workspace1-ranokau/... en todas las rutas (filesystem desincronizado en
# lascar, confirmado 2026-09-17).
#
# No reanudable: es una pasada batch sobre datos fijos, no un csv que sigue
# creciendo -- si corta a mitad, se vuelve a correr entero.
#
# NO usar `source "$VENV/bin/activate"` (bug en lascar, ver
# vllm_server_licencias.sbatch) -- se exporta PATH/VIRTUAL_ENV a mano.

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export CMP_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen2.5-32B-Instruct
export CMP_ENTRADA="salida/comparacion_modelos/subset_urls.jsonl"
export CMP_TENSOR_PARALLEL_SIZE=2
export CMP_GPU_MEMORY_UTILIZATION=0.9
export CMP_MAX_MODEL_LEN=4096
export CMP_TIEMPOS="salida/comparacion_modelos/tiempos.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
