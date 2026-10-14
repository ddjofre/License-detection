#!/bin/bash
#SBATCH --job-name=job1
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# juez1 = juzga la salida de cmp1 (Qwen2.5-32B-Instruct, ver
# slurm/comparar_evaluadores_qwen2.5-32b.sbatch) con Llama-3.3-70B-Instruct
# offline/batch (vLLM, sin servidor, 2 GPUs).
#
# Corre 03_juzgar_licencias_llm.py, que ahora (2026-10-02)
# además del comentario manual de referencia/licencias_original_a_mano.xlsx le pasa al
# juez el diccionario de licencias conocidas (referencia/dicc_lic.json,
# transcripto de referencia/dicc_lic.png) como referencia -- para que pueda chequear si
# el veredicto SI/NO/DEPENDE del evaluador es consistente con lo que esa
# licencia realmente permite, no solo si "suena parecido" al comentario.
#
# Por eso max_model_len subió de 2048 a 4096 en el script (el diccionario
# agrega ~2575 caracteres fijos a cada prompt) -- si tira OOM al cargar
# Llama-3.3-70B en 2 GPUs con este contexto más grande, ver la nota en el
# propio 03_juzgar_licencias_llm.py (primero probar --enforce-eager, si no
# alcanza subir a 3-4 GPUs).
#
# Mismo motivo que vllm_server_licencias.sbatch para la cuenta latamgpt +
# partition=lascar (único combo con GPU para esa cuenta) y el prefijo
# /workspace1-ranokau/... en todas las rutas.
#
# No reanudable: es una pasada batch sobre datos fijos, no un csv que sigue
# creciendo -- si corta a mitad, se vuelve a correr entero.
#
# NO usar `source "$VENV/bin/activate"` (bug en lascar, ver
# vllm_server_licencias.sbatch) -- se exporta PATH/VIRTUAL_ENV a mano.
#
# NO LANZAR TODAVIA -- armado a pedido del usuario para revisar antes de
# correr (2026-10-02).

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export JUZGAR_EVALUADAS="salida/qwen2.5-32b-instruct/licencias_evaluadas_qwen2.5-32b-instruct.csv"
export JUZGAR_SALIDA="salida/qwen2.5-32b-instruct/juicio_llama.csv"

cd "$REPO"

"$PY" 03_juzgar_licencias_llm.py
