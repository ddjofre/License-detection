#!/bin/bash
#SBATCH --job-name=jt3
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# jt3 = juzga la salida de t3 (Qwen3.6-27B a temperatura 0.7, ver
# slurm/comparar_evaluadores_qwen3.6-27b_temp0.7.sbatch) con
# Llama-3.3-70B-Instruct offline/batch (vLLM, sin servidor, 2 GPUs).
# Contraparte de job3 (mismo evaluador, temperatura 0).
#
# OJO: Qwen3.6-27B quedó descartado a temperatura 0 (no sigue el chat
# template, ver RESULTADOS_comparacion_modelos.md) -- se juzga igual por
# consistencia con el resto, pero no se espera que la temperatura cambie el
# problema de fondo.
#
# Salida separada (juicio_llama_temp0.7.csv) para no pisar juicio_llama.csv
# de la corrida a temperatura 0.
#
# Mismo script y mismo motivo que slurm/juzgar_qwen2.5-32b.sbatch (ver ese
# archivo para el detalle completo).

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export JUZGAR_EVALUADAS="salida/qwen3.6-27b/licencias_evaluadas_qwen3.6-27b_temp0.7.csv"
export JUZGAR_SALIDA="salida/qwen3.6-27b/juicio_llama_temp0.7.csv"

cd "$REPO"

"$PY" 03_juzgar_licencias_llm.py
