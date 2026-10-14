#!/bin/bash
#SBATCH --job-name=t2
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# t2 = Qwen2.5-72B-Instruct a CMP_TEMPERATURE=0.7 (contraparte de cmp2 a
# temperatura 0). Ver slurm/comparar_evaluadores_qwen2.5-32b_temp0.7.sbatch
# para el detalle completo.

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export CMP_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen2.5-72B-Instruct
export CMP_ENTRADA="salida/comparacion_modelos/subset_urls.jsonl"
export CMP_TENSOR_PARALLEL_SIZE=4
export CMP_GPU_MEMORY_UTILIZATION=0.9
export CMP_MAX_MODEL_LEN=4096
export CMP_TEMPERATURE=0.7
export CMP_TIEMPOS="salida/comparacion_modelos/tiempos.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
