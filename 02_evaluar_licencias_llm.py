"""Evalúa (y matiza) un subconjunto FIJO de URLs con un modelo vía vLLM en
modo offline/batch -- una sola corrida, sin servidor, mismo patrón que
03_juzgar_licencias_llm.py -- pensada para comparar tiempos y resultados entre
distintas versiones de Qwen sobre el mismo subconjunto.

A diferencia de compartido/evaluar_licencias_modo_server.py (pensado para la corrida de
producción continua, sirviendo un jsonl que crece durante horas/días vía un
servidor vLLM + requests HTTP de a una unidad por vez), acá el conjunto de
URLs a evaluar es chico y se conoce de antemano, así que no tiene sentido
pagar el costo de un servidor aparte ni de ir pidiendo de a un request:
se cargan todos los prompts de evaluación de TODAS las URLs del subconjunto
y se mandan de una sola vez a `generate()`, dejando que vLLM los batchee de
verdad (en vez de ir de a un POST HTTP por vez, que nunca aprovecha el
continuous batching real de vLLM y además mezclaría latencia de red en la
medición de tiempo que es justo lo que se quiere comparar).

Dos pasadas a generate() sobre el MISMO modelo ya cargado en memoria (no se
recarga entre una y otra):
  1. Evaluar: un prompt por unidad de texto (footer/subpágina/PDF) de cada
     URL -- mismo criterio que compartido/evaluar_licencias_modo_server.py para combinar el
     veredicto final de la URL (el más restrictivo de sus unidades gana).
  2. Matizar: un prompt por cada URL cuyo veredicto combinado dio DEPENDE,
     usando el texto de las unidades que individualmente dieron DEPENDE.

Reutiliza de compartido/evaluar_licencias_modo_server.py todo lo que no depende de pegarle a
un servidor por HTTP: los prompts, el parseo de la respuesta, el armado de
unidades por fila, y `combinar_detalle` (la función que decide el veredicto
final de una URL a partir de sus unidades).

No es reanudable ni continuo -- es una sola pasada batch sobre un
subconjunto fijo, igual que 03_juzgar_licencias_llm.py.

Recorte de respaldo por TOKENS reales (no solo caracteres): MAX_CHARS_LICENCIA
(de compartido/evaluar_licencias_modo_server.py) es una heurística en caracteres que asume ~3.5
caracteres por token -- margen que en producción (requests HTTP de a una
unidad) solo hace fallar ESA unidad puntual si se pasa, sin arrastrar nada
más. Acá en cambio `generate()` valida TODOS los prompts de la lista antes de
generar cualquier cosa -- si uno solo se pasa del límite del modelo, se cae
el batch ENTERO, perdiendo el resultado de todas las demás unidades que sí
habrían andado bien. Confirmado en una corrida real (2026-10-02, jobs 13156 y
13158): un prompt de 4097 tokens contra un límite de 4096 tumbó Qwen2.5-72B y
Qwen3-Next-80B-A3B enteros (Qwen2.5-32B y Qwen3.6-27B no lo sufrieron con el
mismo subconjunto -- aparentemente por una diferencia de tokenización entre
familias, no algo determinable de antemano por caracteres). Por eso, antes de
mandar cada prompt a generate(), se lo recorta también por su conteo de
tokens REAL según el tokenizer del modelo que se está por cargar -- más caro
de calcular que el recorte por caracteres, pero se hace una sola vez por
prompt y evita perder una corrida entera por un solo documento largo.

Variables de entorno:
  CMP_MODELO (obligatoria): ruta del checkpoint a comparar.
  CMP_ENTRADA: jsonl con el subconjunto fijo de URLs a evaluar (default:
               el jsonl completo de producción -- para una comparación de
               verdad, armar un subconjunto aparte y apuntar acá).
  CMP_SALIDA: csv de salida (default: salida/<slug del
              modelo>/licencias_evaluadas_<slug del modelo>.csv -- carpeta
              por modelo, mismo patrón que ya usan los sbatch de
              compartido/evaluar_licencias_modo_server.py para qwen32b/qwen72b, PERO con el
              slug además en el nombre del archivo: a propósito, para que
              el csv se siga identificando solo si alguien lo copia o lo
              mueve fuera de su carpeta).
  CMP_TENSOR_PARALLEL_SIZE: GPUs a usar (default 1).
  CMP_GPU_MEMORY_UTILIZATION: default 0.9, igual que el servidor de
              producción de Qwen2.5-32B.
  CMP_MAX_MODEL_LEN: default 4096, igual que compartido/evaluar_licencias_modo_server.py.
  CMP_TEMPERATURE: default 0 (determinístico, igual que compartido/evaluar_licencias_modo_server.py
              y 03_juzgar_licencias_llm.py). Si se sube por encima de 0, el
              mismo modelo puede dar veredictos distintos entre corridas --
              el nombre de archivo de salida por default agrega "_tempX" en
              ese caso (si es 0 queda igual que siempre), para no pisar sin
              querer una corrida con otra temperatura del mismo modelo. La
              temperatura usada queda registrada en CMP_TIEMPOS también.
  CMP_TIEMPOS: csv consolidado donde se agrega una fila por corrida con
              tiempos y conteos (default: comparacion_modelos/tiempos.csv).

Tiene que correr con el mismo venv que tiene vllm instalado (mismo venv que
usan los servidores vllm_server_*.sbatch) -- no hace falta nodo con el
servidor ya levantado, este script levanta el modelo él mismo.
"""

import csv
import importlib
import json
import os
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Reutiliza los prompts/parseo/armado de unidades del evaluador en modo
# servidor (compartido/evaluar_licencias_modo_server.py) -- ahí vive la
# lógica compartida (PROMPT, PROMPT_MATIZ, parsear_respuesta,
# unidades_de_fila, combinar_detalle, etc.), este script solo cambia CÓMO
# se le pregunta al modelo (batch/chat() acá, HTTP uno por uno ahí).
# "compartido." porque es un subpaquete (compartido/__init__.py) -- import
# normal no funciona para los módulos con prefijo numérico (01_, 03_, etc,
# ver el resto del pipeline) ni para éste (vive en un subpaquete, no es
# visible como nombre simple desde la raíz) -- importlib resuelve los dos
# casos, mismo truco que usan las migraciones de Django/Alembic.
ev = importlib.import_module("compartido.evaluar_licencias_modo_server")

csv.field_size_limit(10_000_000)  # mismo motivo que compartido/evaluar_licencias_modo_server.py

MARGEN_SEGURIDAD_TOKENS = 100  # colchón extra además de max_tokens de salida
# (subido de 32 a 100 al pasar de generate() a chat(): el chat template
# agrega marcas de turno/rol que recortar_a_tokens, al medir solo el texto
# plano antes de aplicar el template, no contempla -- ver más abajo)


def leer_jsonl(ruta: Path) -> list[dict]:
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                filas.append(json.loads(linea))
    return filas


def recortar_a_tokens(tokenizer, prompt: str, max_tokens_prompt: int) -> str:
    """Recorte de respaldo por tokens reales -- ver docstring del módulo.
    No hace nada si el prompt ya entra (caso normal); solo actúa sobre los
    pocos prompts donde el recorte por caracteres de MAX_CHARS_LICENCIA no
    alcanzó."""
    ids = tokenizer.encode(prompt)
    if len(ids) <= max_tokens_prompt:
        return prompt
    return tokenizer.decode(ids[:max_tokens_prompt])


def main() -> None:
    t_inicio = time.time()

    modelo = os.environ.get("CMP_MODELO")
    if not modelo:
        raise SystemExit("CMP_MODELO es obligatoria -- ruta del checkpoint a comparar")
    slug = Path(modelo).name.lower()
    temperatura = float(os.environ.get("CMP_TEMPERATURE", "0"))
    # a temperatura 0 el nombre de archivo queda igual que siempre; apenas
    # se sube, se agrega "_tempX" para no pisar sin querer una corrida a
    # otra temperatura del mismo modelo.
    sufijo_temp = "" if temperatura == 0 else f"_temp{temperatura}"

    entrada = Path(os.environ.get("CMP_ENTRADA", str(ev.ENTRADA)))
    salida = Path(os.environ.get(
        "CMP_SALIDA",
        f"salida/{slug}/licencias_evaluadas_{slug}{sufijo_temp}.csv",
    ))
    tensor_parallel_size = int(os.environ.get("CMP_TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = float(os.environ.get("CMP_GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.environ.get("CMP_MAX_MODEL_LEN", "4096"))
    tiempos_csv = Path(os.environ.get(
        "CMP_TIEMPOS", "salida/comparacion_modelos/tiempos.csv"
    ))

    filas = leer_jsonl(entrada)
    comentarios_por_url = ev._comentarios_por_url_csv()
    print(f"[{slug}] {len(filas)} URLs leídas de {entrada}", flush=True)

    # tokenizer solo (sin cargar el modelo todavía) -- liviano, para poder
    # recortar los prompts por tokens reales antes de levantar vLLM.
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    max_tokens_prompt_eval = max_model_len - 300 - MARGEN_SEGURIDAD_TOKENS
    max_tokens_prompt_matiz = max_model_len - 200 - MARGEN_SEGURIDAD_TOKENS

    # --- armar los prompts de evaluación, una unidad de texto a la vez ---
    prompts_eval = []
    indices_eval = []  # (fila_idx, unidad) por cada prompt, mismo orden que prompts_eval
    recortados_por_token = 0
    for fila_idx, fila in enumerate(filas):
        for unidad in ev.unidades_de_fila(fila, comentarios_por_url):
            texto = ev.texto_de_unidad(unidad).strip()
            if not texto:
                continue
            prompt = ev.PROMPT.format(licencia=texto[:ev.MAX_CHARS_LICENCIA])
            prompt_recortado = recortar_a_tokens(tokenizer, prompt, max_tokens_prompt_eval)
            if prompt_recortado is not prompt:
                recortados_por_token += 1
            prompts_eval.append(prompt_recortado)
            indices_eval.append((fila_idx, unidad))

    print(
        f"[{slug}] {len(prompts_eval)} unidades de texto a evaluar "
        f"({recortados_por_token} recortadas por exceder tokens del modelo)",
        flush=True,
    )

    t_carga_inicio = time.time()
    llm = LLM(
        model=modelo,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    t_carga = time.time() - t_carga_inicio

    sampling_eval = SamplingParams(temperature=temperatura, max_tokens=300)
    sampling_matiz = SamplingParams(temperature=temperatura, max_tokens=200)

    # chat() en vez de generate(): aplica el chat template del modelo (marca
    # de turno de usuario, system, etc.) -- generate() manda el prompt como
    # texto plano, sin esas marcas, y el modelo puede terminar CONTINUANDO
    # el texto en vez de responder como asistente (confirmado con
    # Qwen3-Next-80B-A3B en 04_asignar_licencia_llm.py: 56% de las respuestas
    # no seguían el formato pedido, porque el modelo seguía escribiendo el
    # texto de la página en vez de contestar).
    conversaciones_eval = [[{"role": "user", "content": p}] for p in prompts_eval]

    t_eval_inicio = time.time()
    salidas_eval = llm.chat(conversaciones_eval, sampling_eval) if conversaciones_eval else []
    t_evaluar = time.time() - t_eval_inicio

    # --- reconstruir el detalle de cada URL a partir de las salidas, en orden ---
    detalle_por_fila: list[list[dict]] = [[] for _ in filas]
    for (fila_idx, unidad), salida_llm in zip(indices_eval, salidas_eval):
        veredicto, justificacion = ev.parsear_respuesta(salida_llm.outputs[0].text)
        detalle_por_fila[fila_idx].append({
            "categoria": unidad.get("categoria"),
            "via": unidad.get("via"),
            "url_origen": unidad.get("url_origen"),
            "markdown_path": unidad.get("markdown_path"),
            "texto_inline": unidad.get("texto_inline"),
            "veredicto": veredicto,
            "justificacion": justificacion,
        })

    resultados: list[tuple[str, str, list[dict]]] = []
    for detalle in detalle_por_fila:
        if not detalle:
            resultados.append(
                ("SIN_TEXTO", "No se detectó texto de licencia/términos/privacidad en el sitio.", [])
            )
        else:
            veredicto, justificacion = ev.combinar_detalle(detalle)
            resultados.append((veredicto, justificacion, detalle))

    # --- armar los prompts de matiz, solo para las URLs que dieron DEPENDE ---
    prompts_matiz = []
    indices_matiz = []  # fila_idx por cada prompt, mismo orden que prompts_matiz
    recortados_por_token_matiz = 0
    for fila_idx, (veredicto, justificacion, detalle) in enumerate(resultados):
        if veredicto != "DEPENDE":
            continue
        texto_depende = ev.texto_depende_de(detalle)
        if texto_depende:
            prompt = ev.PROMPT_MATIZ.format(
                licencia=texto_depende[:ev.MAX_CHARS_LICENCIA],
                justificacion_llm=justificacion,
            )
            prompt_recortado = recortar_a_tokens(tokenizer, prompt, max_tokens_prompt_matiz)
            if prompt_recortado is not prompt:
                recortados_por_token_matiz += 1
            prompts_matiz.append(prompt_recortado)
            indices_matiz.append(fila_idx)

    print(
        f"[{slug}] {len(prompts_matiz)} URLs DEPENDE a matizar "
        f"({recortados_por_token_matiz} recortadas por exceder tokens del modelo)",
        flush=True,
    )

    conversaciones_matiz = [[{"role": "user", "content": p}] for p in prompts_matiz]

    t_matiz_inicio = time.time()
    salidas_matiz = llm.chat(conversaciones_matiz, sampling_matiz) if conversaciones_matiz else []
    t_matizar = time.time() - t_matiz_inicio

    niveles: dict[int, tuple[str, str]] = {
        fila_idx: ev.parsear_nivel(salida_llm.outputs[0].text)
        for fila_idx, salida_llm in zip(indices_matiz, salidas_matiz)
    }

    # --- escribir el csv de evaluadas, mismo formato que compartido/evaluar_licencias_modo_server.py ---
    conteo_veredicto: dict[str, int] = {}
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=ev.CAMPOS_SALIDA)
        escritor.writeheader()
        for fila_idx, fila in enumerate(filas):
            veredicto, justificacion, detalle = resultados[fila_idx]
            conteo_veredicto[veredicto] = conteo_veredicto.get(veredicto, 0) + 1
            if veredicto == "DEPENDE":
                nivel, explicacion_nivel = niveles.get(
                    fila_idx, ("", "Sin texto de licencia para matizar.")
                )
            else:
                nivel, explicacion_nivel = "", ""
            escritor.writerow({
                "url": fila["url"],
                "licencia": fila.get("licencia", ""),
                "veredicto_llm": veredicto,
                "justificacion_llm": justificacion,
                "respuesta_cruda_llm": json.dumps(detalle, ensure_ascii=False),
                "nivel_uso_llm": nivel,
                "explicacion_nivel_llm": explicacion_nivel,
            })

    t_total = time.time() - t_inicio
    print(f"[{slug}] listo, {len(filas)} URLs evaluadas -> {salida}", flush=True)
    print(f"[{slug}] conteo de veredicto_llm: {conteo_veredicto}", flush=True)
    print(
        f"[{slug}] tiempos -- carga modelo: {t_carga:.1f}s, evaluar: {t_evaluar:.1f}s, "
        f"matizar: {t_matizar:.1f}s, total: {t_total:.1f}s",
        flush=True,
    )

    # --- agregar una fila al csv consolidado de tiempos, para comparar entre corridas ---
    campos_tiempos = [
        "modelo", "temperatura", "gpus", "urls", "unidades_evaluadas", "urls_matizadas",
        "tiempo_carga_s", "tiempo_evaluar_s", "tiempo_matizar_s", "tiempo_total_s",
        "veredictos",
    ]
    es_nuevo_tiempos = not tiempos_csv.exists()
    tiempos_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(tiempos_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos_tiempos)
        if es_nuevo_tiempos:
            escritor.writeheader()
        escritor.writerow({
            "modelo": slug,
            "temperatura": temperatura,
            "gpus": tensor_parallel_size,
            "urls": len(filas),
            "unidades_evaluadas": len(prompts_eval),
            "urls_matizadas": len(prompts_matiz),
            "tiempo_carga_s": f"{t_carga:.1f}",
            "tiempo_evaluar_s": f"{t_evaluar:.1f}",
            "tiempo_matizar_s": f"{t_matizar:.1f}",
            "tiempo_total_s": f"{t_total:.1f}",
            "veredictos": json.dumps(conteo_veredicto, ensure_ascii=False),
        })
    print(f"[{slug}] tiempos registrados -> {tiempos_csv}", flush=True)


if __name__ == "__main__":
    main()
