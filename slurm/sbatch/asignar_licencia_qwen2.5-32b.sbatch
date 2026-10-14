#!/bin/bash
#SBATCH --job-name=asig1
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# asig1 = asigna licencia (04_asignar_licencia_llm.py) con
# Qwen2.5-32B-Instruct offline/batch (vLLM, sin servidor, 2 GPUs) -- mismas
# GPUs que cmp1 (slurm/comparar_evaluadores_qwen2.5-32b.sbatch), mismo
# checkpoint.
#
# Corre sobre las 467 URLs COMPLETAS de licencias_detectadas.csv (sin
# ASIG_SUBSET) -- a diferencia de la comparación de evaluadores, acá SÍ se
# incluyen las 126 URLs "Alianza" (a pedido del usuario): también
# necesitan una licencia asignada, aunque no tengan un veredicto SI/NO/DEPENDE
# calculado (esa parte sí se hizo solo sobre las 341 no-alianza).
#
# NO LANZAR TODAVIA -- armado a pedido del usuario para revisar antes de
# correr (2026-10-05).
#
# NO usar `source "$VENV/bin/activate"` (bug en lascar, ver
# vllm_server_licencias.sbatch) -- se exporta PATH/VIRTUAL_ENV a mano.

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export ASIG_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen2.5-32B-Instruct
export ASIG_ENTRADA="salida/licencias_detectadas.csv"
export ASIG_TENSOR_PARALLEL_SIZE=2
export ASIG_GPU_MEMORY_UTILIZATION=0.9
export ASIG_MAX_MODEL_LEN=4096
export ASIG_TEMPERATURE=0
export ASIG_TIEMPOS="salida/comparacion_modelos/tiempos_asignacion.csv"

cd "$REPO"

"$PY" 04_asignar_licencia_llm.py
