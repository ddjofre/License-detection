#!/bin/bash
#SBATCH --job-name=jt2
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# jt2 = juzga la salida de t2 (Qwen2.5-72B-Instruct a temperatura 0.7, ver
# slurm/comparar_evaluadores_qwen2.5-72b_temp0.7.sbatch) con
# Llama-3.3-70B-Instruct offline/batch (vLLM, sin servidor, 2 GPUs -- el
# juez mismo siempre corre a temperature=0, hardcodeado en
# 03_juzgar_licencias_llm.py, no es configurable). Contraparte de job2 (mismo
# evaluador, temperatura 0).
#
# Salida separada (juicio_llama_temp0.7.csv) para no pisar juicio_llama.csv
# de la corrida a temperatura 0.
#
# Mismo script y mismo motivo que slurm/juzgar_qwen2.5-32b.sbatch (ver ese
# archivo para el detalle completo: diccionario de licencias, max_model_len,
# chat template, recorte por tokens).

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export JUZGAR_EVALUADAS="salida/qwen2.5-72b-instruct/licencias_evaluadas_qwen2.5-72b-instruct_temp0.7.csv"
export JUZGAR_SALIDA="salida/qwen2.5-72b-instruct/juicio_llama_temp0.7.csv"

cd "$REPO"

"$PY" 03_juzgar_licencias_llm.py
