#!/bin/bash
#SBATCH --job-name=at3
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# at3 = asigna licencia con Qwen3.6-27B a ASIG_TEMPERATURE=0.7 (contraparte
# de asig3). OJO: Qwen3.6-27B quedó descartado a temperatura 0 -- falla el
# 98.5% de las veces (no sigue el chat template, ver
# RESULTADOS_comparacion_modelos.md) -- se corre igual por consistencia,
# sin esperar que la temperatura arregle el problema de fondo.
#
# Ver slurm/asignar_licencia_qwen2.5-32b_temp0.7.sbatch para el detalle
# completo.

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export ASIG_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen3.6-27B
export ASIG_ENTRADA="salida/licencias_detectadas.csv"
export ASIG_TENSOR_PARALLEL_SIZE=2
export ASIG_GPU_MEMORY_UTILIZATION=0.9
export ASIG_MAX_MODEL_LEN=4096
export ASIG_TEMPERATURE=0.7
export ASIG_TIEMPOS="salida/comparacion_modelos/tiempos_asignacion.csv"

cd "$REPO"

"$PY" 04_asignar_licencia_llm.py
