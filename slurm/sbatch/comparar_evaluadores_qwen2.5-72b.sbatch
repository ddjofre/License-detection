#!/bin/bash
#SBATCH --job-name=cmp2
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# cmp2 = Qwen2.5-72B-Instruct. Ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch
# para el mapeo completo de job-names cortos (cmp1=32B, cmp2=72B, cmp3=Qwen3.6-27B,
# cmp4=Qwen3-Next-80B-A3B).
#
# Corre 02_evaluar_licencias_llm.py con Qwen2.5-72B-Instruct
# offline/batch (vLLM, sin servidor, 4 GPUs -- mismo --tensor-parallel-size
# que ya usa vllm_server_qwen72b.sbatch: 135.4 GiB bf16, confirmado via
# model.safetensors.index.json) sobre el mismo subconjunto fijo de 341 URLs
# que slurm/comparar_evaluadores_qwen2.5-32b.sbatch.
#
# Ver ese script para el detalle completo de la comparación (subconjunto,
# carpeta de salida por modelo, tiempos.csv compartido, motivo de la cuenta
# latamgpt/partition=lascar/prefijo -ranokau).
#
# Salida: salida/qwen2.5-72b-instruct/licencias_evaluadas_qwen2.5-72b-instruct.csv
# (slug del modelo en la carpeta Y en el nombre del archivo, a propósito --
# ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch)

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
export CMP_TIEMPOS="salida/comparacion_modelos/tiempos.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
