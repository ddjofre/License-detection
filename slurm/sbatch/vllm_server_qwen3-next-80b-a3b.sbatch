#!/bin/bash
#SBATCH --job-name=job5
#SBATCH --partition=lascar
#SBATCH --account=latamgpt
#SBATCH --qos=lascar-unlimit
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.out
#SBATCH --error=/workspace1-ranokau/users/ddjofre/licencias/License-detection/slurm/logs/%x_%j.err

# Levanta Qwen3-Next-80B-A3B-Instruct (el ganador de la comparación de
# modelos, ver docs/RESULTADOS_comparacion_modelos.md) como servidor vLLM
# para la corrida de PRODUCCIÓN de
# compartido/evaluar_licencias_modo_server.py -- con el matizado ahora
# enriquecido con el contexto de referencia/excepciones_latam_v1.csv (ver
# PROMPT_MATIZ / bloque_contexto_pais en ese módulo).
#
# Mismo sizing que slurm/sbatch/comparar_evaluadores_qwen3-next-80b-a3b.sbatch
# (4 GPUs, tensor-parallel-size=4, max-model-len=4096) -- ver ese archivo
# para la justificación completa (checkpoint 151.5 GiB bf16,
# num_attention_heads=16 solo divisible por 1/2/4/8/16).
#
# Mismo patrón que slurm/vllm_server_licencias.sbatch del repo original:
# cuenta latamgpt + partition=lascar (único combo válido para GPU en esa
# cuenta), prefijo /workspace1-ranokau/... en TODAS las rutas (lascar tiene
# su propio /workspace1 local desincronizado) y NO usar
# "source $VENV/bin/activate" (VIRTUAL_ENV queda hardcodeado sin el
# prefijo -ranokau y cae en silencio al Python de sistema).
#
# El puerto se elige libre al arrancar y se anuncia en
# slurm/logs/vllm_endpoint_job5_<jobid>.txt (host + puerto) -- el evaluador
# (slurm/sbatch/evaluar_licencias_qwen3-next-80b-a3b.sbatch) lee ese glob.
# Mismo partition=lascar + qos=lascar-unlimit que el evaluador, por lo que
# cae en el mismo nodo sin necesitar --nodelist (confirmado en corridas
# anteriores -- el tráfico TCP entre nodos distintos está bloqueado).
#
# Queda corriendo hasta el time limit o hasta cancelarlo a mano (scancel)
# una vez que el evaluador termine -- no se apaga solo.

set -euo pipefail

REPO=/workspace1-ranokau/users/ddjofre/licencias/License-detection
VENV=/workspace1-ranokau/users/ddjofre/venv_licencias
MODELO=/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen3-Next-80B-A3B-Instruct
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
PY="$VENV/bin/python3"
VLLM="$VENV/bin/vllm"

PORT=$("$PY" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "host=$(hostname) port=$PORT" | tee "$REPO/slurm/logs/vllm_endpoint_job5_${SLURM_JOB_ID}.txt"

"$VLLM" serve "$MODELO" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
