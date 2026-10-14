#!/bin/bash
#SBATCH --job-name=cmp4
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# cmp4 = Qwen3-Next-80B-A3B-Instruct. Ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch
# para el mapeo completo de job-names cortos (cmp1=32B, cmp2=72B, cmp3=Qwen3.6-27B,
# cmp4=Qwen3-Next-80B-A3B).
#
# Corre 02_evaluar_licencias_llm.py con
# Qwen3-Next-80B-A3B-Instruct offline/batch (vLLM, sin servidor, 4 GPUs)
# sobre el mismo subconjunto fijo de 341 URLs que
# slurm/comparar_evaluadores_qwen2.5-32b.sbatch.
#
# Por qué 4 GPUs y no 2: el checkpoint pesa 151.5 GiB bf16 (confirmado via
# model.safetensors.index.json) -- con 2 GPUs A100 (~79 GiB útiles c/u,
# ~158 GiB totales) quedarían apenas ~6 GiB libres en total para el KV
# cache, insuficiente. Con 4 GPUs (~316 GiB totales) quedan ~164 GiB
# libres, margen real.
# Por qué no 3: --tensor-parallel-size tiene que dividir num_attention_heads
# (16 en este modelo, confirmado via config.json) -- 3 no divide 16, vLLM
# no arranca con ese valor. Los únicos tensor-parallel-size válidos para
# este modelo son 1/2/4/8/16.
#
# Es MoE (A3B: ~3B parámetros activos por token de los 80B totales) -- pesa
# más en disco que Qwen2.5-72B pero debería generar bastante más rápido en
# el tiempo de evaluar/matizar, pese a necesitar las mismas 4 GPUs para
# cargar los pesos. Es justo el contraste más interesante de esta
# comparación (ver slurm/comparar_evaluadores_qwen2.5-72b.sbatch).
#
# Ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch para el detalle completo
# de la comparación (subconjunto, carpeta de salida por modelo, tiempos.csv
# compartido, motivo de la cuenta latamgpt/partition=lascar/prefijo -ranokau).
#
# Salida: salida/qwen3-next-80b-a3b-instruct/licencias_evaluadas_qwen3-next-80b-a3b-instruct.csv
# (slug del modelo en la carpeta Y en el nombre del archivo, a propósito --
# ver slurm/comparar_evaluadores_qwen2.5-32b.sbatch)

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export CMP_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen3-Next-80B-A3B-Instruct
export CMP_ENTRADA="salida/comparacion_modelos/subset_urls.jsonl"
export CMP_TENSOR_PARALLEL_SIZE=4
export CMP_GPU_MEMORY_UTILIZATION=0.9
export CMP_MAX_MODEL_LEN=4096
export CMP_TIEMPOS="salida/comparacion_modelos/tiempos.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
