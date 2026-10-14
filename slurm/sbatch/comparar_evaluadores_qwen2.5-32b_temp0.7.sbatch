#!/bin/bash
#SBATCH --job-name=t1
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# t1 = Qwen2.5-32B-Instruct a CMP_TEMPERATURE=0.7 (segunda versión del
# experimento, contraparte de cmp1 que corrió a temperatura 0 -- ver
# slurm/comparar_evaluadores_qwen2.5-32b.sbatch). Mismo input (subset_urls.jsonl,
# 341 URLs), mismas GPUs, mismo script.
#
# A diferencia de cmp1, acá el resultado NO es determinístico -- correrlo de
# nuevo puede dar veredictos distintos para la misma URL. El nombre de
# archivo de salida agrega automáticamente "_temp0.7" (02_evaluar_licencias_llm.py
# lo hace solo cuando CMP_TEMPERATURE != 0), así que no pisa el resultado de
# cmp1 -- quedan los dos lado a lado en la misma carpeta del modelo.
#
# Ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch para el resto del detalle
# (motivo de la cuenta latamgpt/partition=lascar/prefijo -ranokau, por qué
# no reanudable, etc.)

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
export CMP_TEMPERATURE=0.7
export CMP_TIEMPOS="salida/comparacion_modelos/tiempos.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
