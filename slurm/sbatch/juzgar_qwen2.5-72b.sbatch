#!/bin/bash
#SBATCH --job-name=job2
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# juez2 = juzga la salida de cmp2 (Qwen2.5-72B-Instruct, ver
# slurm/comparar_evaluadores_qwen2.5-72b.sbatch) con Llama-3.3-70B-Instruct
# offline/batch (vLLM, sin servidor, 2 GPUs).
#
# Mismo script y mismo motivo que slurm/juzgar_qwen2.5-32b.sbatch (ver ese
# archivo para el detalle completo: diccionario de licencias, max_model_len,
# fallback si tira OOM) -- esta copia solo cambia JUZGAR_EVALUADAS/
# JUZGAR_SALIDA para apuntar a la carpeta de Qwen2.5-72B.
#
# NO usar `source "$VENV/bin/activate"` (bug en lascar, ver
# vllm_server_licencias.sbatch) -- se exporta PATH/VIRTUAL_ENV a mano.

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export JUZGAR_EVALUADAS="salida/qwen2.5-72b-instruct/licencias_evaluadas_qwen2.5-72b-instruct.csv"
export JUZGAR_SALIDA="salida/qwen2.5-72b-instruct/juicio_llama.csv"

cd "$REPO"

"$PY" 03_juzgar_licencias_llm.py
