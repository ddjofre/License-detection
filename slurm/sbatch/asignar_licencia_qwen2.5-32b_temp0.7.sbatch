#!/bin/bash
#SBATCH --job-name=at1
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# at1 = asigna licencia con Qwen2.5-32B-Instruct a ASIG_TEMPERATURE=0.7
# (contraparte de asig1, mismo modelo a temperatura 0). Mismas GPUs, mismo
# input (467 URLs completas, incluidas las "Alianza" -- sin ASIG_SUBSET).
#
# El nombre de archivo de salida agrega automáticamente "_temp0.7"
# (04_asignar_licencia_llm.py lo hace solo cuando ASIG_TEMPERATURE != 0), así
# que no pisa el resultado de asig1 -- quedan los dos lado a lado en la
# carpeta del modelo.
#
# Ver slurm/asignar_licencia_qwen2.5-32b.sbatch para el resto del detalle
# (chat template, recorte por tokens, motivo de la cuenta latamgpt/
# partition=lascar/prefijo -ranokau).

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
export ASIG_TEMPERATURE=0.7
export ASIG_TIEMPOS="salida/comparacion_modelos/tiempos_asignacion.csv"

cd "$REPO"

"$PY" 04_asignar_licencia_llm.py
