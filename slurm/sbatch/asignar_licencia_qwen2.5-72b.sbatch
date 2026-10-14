#!/bin/bash
#SBATCH --job-name=asig2
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# asig2 = asigna licencia con Qwen2.5-72B-Instruct offline/batch (vLLM, sin
# servidor, 4 GPUs) -- mismas GPUs que cmp2. Ver
# slurm/asignar_licencia_qwen2.5-32b.sbatch para el detalle completo
# (subconjunto, por qué 4096/temperatura 0, etc.)
#
# NO LANZAR TODAVIA -- armado a pedido del usuario para revisar antes de
# correr (2026-10-05).

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export ASIG_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen2.5-72B-Instruct
export ASIG_ENTRADA="salida/licencias_detectadas.csv"
# Sin ASIG_SUBSET a propósito: corre sobre las 467 URLs completas, incluidas
# las 126 "Alianza" -- ver slurm/asignar_licencia_qwen2.5-32b.sbatch.
export ASIG_TENSOR_PARALLEL_SIZE=4
export ASIG_GPU_MEMORY_UTILIZATION=0.9
export ASIG_MAX_MODEL_LEN=4096
export ASIG_TEMPERATURE=0
export ASIG_TIEMPOS="salida/comparacion_modelos/tiempos_asignacion.csv"

cd "$REPO"

"$PY" 04_asignar_licencia_llm.py
