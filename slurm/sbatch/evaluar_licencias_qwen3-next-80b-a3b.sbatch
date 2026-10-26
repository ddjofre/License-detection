#!/bin/bash
#SBATCH --job-name=job6
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# Corre compartido/evaluar_licencias_modo_server.py (evaluador de
# PRODUCCIÓN, resumible por url) contra el servidor vLLM de
# slurm/sbatch/vllm_server_qwen3-next-80b-a3b.sbatch -- requiere que ese
# job ya esté corriendo. No pide GPU: solo hace requests HTTP al servidor.
#
# Primera corrida de producción con Qwen3-Next-80B-A3B-Instruct (el
# ganador de la comparación, ver docs/RESULTADOS_comparacion_modelos.md) Y
# con el matizado enriquecido con referencia/excepciones_latam_v1.csv
# (contexto legal por país de origen, agregado solo cuando el veredicto da
# DEPENDE y el dominio matchea -- ver PROMPT_MATIZ en
# compartido/evaluar_licencias_modo_server.py).
#
# EVAL_SALIDA apunta a un archivo NUEVO (no al salida/licencias_evaluadas.csv
# existente, que ya tiene ~441 URLs resueltas con Qwen2.5-32B y el prompt
# viejo sin contexto de país) -- si apuntara al archivo viejo, la lógica de
# resumible por url saltearía esas ~441 URLs y el modelo nuevo + el
# contexto nuevo nunca las tocarían.
#
# Reanudable por url, corre en loop indefinido siguiendo
# salida/licencias_detectadas.jsonl -- si corta por el time limit, un
# sbatch de nuevo retoma donde quedó.
#
# NO usar "source $VENV/bin/activate" (mismo bug que el servidor).

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"

export EVAL_MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen3-Next-80B-A3B-Instruct
export EVAL_SALIDA="salida/licencias_evaluadas_qwen3-next-80b-a3b.csv"
export EVAL_ENDPOINT_GLOB="slurm/logs/vllm_endpoint_job5_*.txt"

cd "$REPO"

"$PY" compartido/evaluar_licencias_modo_server.py
