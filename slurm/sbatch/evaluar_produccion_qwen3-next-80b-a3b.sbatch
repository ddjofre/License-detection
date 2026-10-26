#!/bin/bash
#SBATCH --job-name=job7
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# Corre 02_evaluar_licencias_llm.py (offline/batch, UN solo job -- carga el
# modelo, evalúa y matiza TODO de una, sin servidor aparte) con
# Qwen3-Next-80B-A3B-Instruct (el ganador de la comparación, ver
# docs/RESULTADOS_comparacion_modelos.md) sobre el jsonl COMPLETO de
# producción -- a diferencia de slurm/sbatch/comparar_evaluadores_qwen3-next-80b-a3b.sbatch,
# que apunta a un subconjunto fijo de 341 URLs para comparar modelos entre
# sí, acá NO se fija CMP_ENTRADA, así que toma el default de
# 02_evaluar_licencias_llm.py (ev.ENTRADA = salida/licencias_detectadas.jsonl,
# las 467 URLs completas).
#
# Ya incluye el matizado con contexto de referencia/excepciones_latam_v1.csv
# (ver PROMPT_MATIZ en compartido/evaluar_licencias_modo_server.py) y el
# recorte por tokens reales antes de generate() -- este modo batch, a
# diferencia del servidor de producción, valida TODOS los prompts antes de
# generar nada, así que un solo prompt largo (ver el caso ERROR de
# eppetroecuador.ec en la corrida anterior vía servidor) queda cubierto acá.
#
# Mismo sizing que comparar_evaluadores_qwen3-next-80b-a3b.sbatch (4 GPUs,
# tensor-parallel-size=4, max-model-len=4096 -- checkpoint 151.5 GiB bf16,
# num_attention_heads=16 solo divisible por 1/2/4/8/16) y mismo --time=30min
# (con 341 URLs tardó ~500s, casi todo carga del modelo -- evaluar+matizar
# real tardó ~10-20s; con 467 URLs hay margen de sobra).
#
# CMP_TIEMPOS apunta a salida/produccion/tiempos.csv (no al
# comparacion_modelos/tiempos.csv de la comparación de 4 modelos) para no
# mezclar esta corrida de producción con esas métricas de comparación.
#
# Salida: salida/produccion/qwen3-next-80b-a3b-instruct/licencias_evaluadas_qwen3-next-80b-a3b-instruct.csv
# (CMP_SALIDA se fija a mano más abajo -- el default de 02_evaluar_licencias_llm.py
# pisaría el csv de 341 URLs que ya dejó la comparación de 4 modelos en
# salida/qwen3-next-80b-a3b-instruct/).

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export CMP_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen3-Next-80B-A3B-Instruct
export CMP_TENSOR_PARALLEL_SIZE=4
export CMP_GPU_MEMORY_UTILIZATION=0.9
export CMP_MAX_MODEL_LEN=4096
export CMP_TIEMPOS="salida/produccion/tiempos.csv"
# El default de CMP_SALIDA (salida/<slug>/licencias_evaluadas_<slug>.csv)
# pisaría el csv que ya existe ahí de la comparación de 4 modelos (341
# URLs, job cmp4) -- esta corrida es otra cosa (467 URLs, producción), así
# que va a su propia carpeta.
export CMP_SALIDA="salida/produccion/qwen3-next-80b-a3b-instruct/licencias_evaluadas_qwen3-next-80b-a3b-instruct.csv"

cd "$REPO"

"$PY" 02_evaluar_licencias_llm.py
