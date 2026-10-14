"""Asigna, por URL, la licencia del diccionario (dicc_lic.json
-- 20 licencias conocidas, transcripto de dicc_lic.png) que mejor corresponda
al texto legal que encontró el detector (comentarios_licencia de
licencias_detectadas.csv: licencia/términos y condiciones/política de
privacidad/datos, lo que haya). Modo offline/batch (vLLM, sin servidor, un
modelo por corrida) -- mismo patrón que 02_evaluar_licencias_llm.py, para
poder correr los mismos 4 Qwen de esa comparación también en esta tarea.

A diferencia de compartido/evaluar_licencias_modo_server.py/02_evaluar_licencias_llm.py (que
evalúan UNIDAD por unidad y combinan el veredicto), acá se manda el texto
COMPLETO de la URL en un solo prompt (un prompt por URL, no por unidad):
asignar una licencia es una lectura holística del texto encontrado, no algo
donde "la unidad más restrictiva gana" tenga sentido como criterio de
combinación. El texto se saca directo de la columna comentarios_licencia del
csv del detector (ya viene combinada y recortada a 8000 caracteres por
01_detectar_licencia.py), no de unidades por separado.

Reutiliza el diccionario de licencias ya cargado por 03_juzgar_licencias_llm.py
(DICCIONARIO_LICENCIAS_TXT, DICC_LICENCIAS) y el recorte de respaldo por
tokens reales de 02_evaluar_licencias_llm.py (recortar_a_tokens) -- mismo
motivo que ahí: en batch, un solo prompt que se pase del límite del modelo
tumba TODO el job (confirmado con Qwen2.5-72B/Qwen3-Next-80B-A3B en la
comparación de evaluadores, ver PLAN_comparacion_modelos.md).

Variables de entorno (mismo patrón que 02_evaluar_licencias_llm.py):
  ASIG_MODELO (obligatoria): ruta del checkpoint.
  ASIG_ENTRADA: csv de licencias detectadas a procesar (default:
                salida/licencias_detectadas.csv).
  ASIG_SUBSET: si se pasa, un csv/jsonl con una columna/campo "url" que
                acota qué URLs de ASIG_ENTRADA procesar (default: sin acotar,
                procesa todas las filas de ASIG_ENTRADA). Para reusar el
                mismo subconjunto de 341 URLs de la comparación de
                evaluadores, apuntar a
                salida/comparacion_modelos/subset_urls.csv
                -- OJO que ese subconjunto excluye las URLs "Alianza", que
                probablemente también necesiten una licencia asignada; queda
                a criterio de quien corra esto.
  ASIG_SALIDA: csv de salida (default: salida/<slug del
                modelo>/licencia_asignada_<slug del modelo>.csv -- mismo
                patrón de carpeta-por-modelo + slug en el archivo que
                02_evaluar_licencias_llm.py).
  ASIG_TENSOR_PARALLEL_SIZE / ASIG_GPU_MEMORY_UTILIZATION / ASIG_MAX_MODEL_LEN:
                mismos defaults que 02_evaluar_licencias_llm.py (1, 0.9, 4096).
  ASIG_TEMPERATURE: default 0.
  ASIG_TIEMPOS: csv consolidado de tiempos (default:
                comparacion_modelos/tiempos_asignacion.csv -- separado del
                tiempos.csv de evaluar, es otra tarea).

No es reanudable ni continuo -- una sola pasada batch, igual que
02_evaluar_licencias_llm.py.
"""

import csv
import importlib
import json
import os
import re
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# import normal no funciona: el nombre del módulo empieza con dígito
# (prefijo numérico del pipeline) -- ver 02_evaluar_licencias_llm.py.
recortar_a_tokens = importlib.import_module("02_evaluar_licencias_llm").recortar_a_tokens
_juez = importlib.import_module("03_juzgar_licencias_llm")
DICC_LICENCIAS = _juez.DICC_LICENCIAS
DICCIONARIO_LICENCIAS_TXT = _juez.DICCIONARIO_LICENCIAS_TXT

csv.field_size_limit(10_000_000)  # mismo motivo que compartido/evaluar_licencias_modo_server.py

MAX_CHARS_LICENCIA = 5000  # mismo margen que compartido/evaluar_licencias_modo_server.py
MARGEN_SEGURIDAD_TOKENS = 100  # colchón para el chat template + salida

with open(DICC_LICENCIAS, encoding="utf-8") as _f:
    LICENCIAS_VALIDAS = list(json.load(_f).keys()) + ["Ninguna coincide"]
_PATRON_LICENCIA = "|".join(re.escape(l) for l in LICENCIAS_VALIDAS)

PROMPT_ASIGNAR = """Eres un asistente legal especializado en licencias de datos. Te paso el texto \
legal encontrado en el sitio de origen de un contenido -- puede ser una licencia, \
términos y condiciones de uso, política de privacidad/datos, u otro texto legal \
del sitio.

Tu tarea es asignar la licencia que MEJOR corresponda de esta lista de licencias \
conocidas, según lo que el texto REALMENTE permite y restringe (no solo por el \
nombre que aparezca literal, si es que aparece):

{diccionario_licencias}

Si el texto no corresponde claramente a ninguna de estas licencias (ej. es solo \
una política de privacidad sin mención de licencia de datos, o el texto es \
insuficiente o ambiguo), respondé "Ninguna coincide".

Responde SOLO en este formato:
LICENCIA: <nombre EXACTO de la lista de arriba, o "Ninguna coincide">
JUSTIFICACION: <una o dos frases citando la cláusula relevante>

Texto encontrado:
{texto}
"""

CAMPOS_SALIDA = [
    "url",
    "licencia_detectada",
    "licencia_asignada_llm",
    "justificacion_asignacion_llm",
]


def parsear_asignacion(texto: str) -> tuple[str, str]:
    m_l = re.search(rf"LICENCIA:\s*({_PATRON_LICENCIA})", texto, re.IGNORECASE)
    m_j = re.search(r"JUSTIFICACION:\s*(.+)", texto, re.IGNORECASE | re.DOTALL)
    if m_l:
        crudo = m_l.group(1)
        # normaliza a la capitalización exacta del diccionario
        licencia = next((l for l in LICENCIAS_VALIDAS if l.lower() == crudo.lower()), crudo)
    else:
        # el modelo no devolvió ninguna de las licencias conocidas -- se
        # guarda la línea cruda en vez de descartarla en silencio, para
        # poder auditar por qué no matcheó (typo del modelo, formato
        # distinto, etc.)
        m_crudo = re.search(r"LICENCIA:\s*(.+)", texto)
        licencia = f"(sin match) {m_crudo.group(1).strip()}" if m_crudo else "(sin match) " + texto.strip()[:200]
    justificacion = m_j.group(1).strip() if m_j else texto.strip()
    return licencia, justificacion


def leer_urls_subset(ruta: Path) -> set[str]:
    if ruta.suffix.lower() == ".jsonl":
        urls = set()
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    urls.add(json.loads(linea)["url"])
        return urls
    with open(ruta, newline="", encoding="utf-8") as f:
        return {fila["url"] for fila in csv.DictReader(f) if fila.get("url")}


def main() -> None:
    t_inicio = time.time()

    modelo = os.environ.get("ASIG_MODELO")
    if not modelo:
        raise SystemExit("ASIG_MODELO es obligatoria -- ruta del checkpoint a usar")
    slug = Path(modelo).name.lower()
    temperatura = float(os.environ.get("ASIG_TEMPERATURE", "0"))
    sufijo_temp = "" if temperatura == 0 else f"_temp{temperatura}"

    entrada = Path(os.environ.get("ASIG_ENTRADA", "salida/licencias_detectadas.csv"))
    subset_path = os.environ.get("ASIG_SUBSET")
    salida = Path(os.environ.get(
        "ASIG_SALIDA",
        f"salida/{slug}/licencia_asignada_{slug}{sufijo_temp}.csv",
    ))
    tensor_parallel_size = int(os.environ.get("ASIG_TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization = float(os.environ.get("ASIG_GPU_MEMORY_UTILIZATION", "0.9"))
    max_model_len = int(os.environ.get("ASIG_MAX_MODEL_LEN", "4096"))
    tiempos_csv = Path(os.environ.get(
        "ASIG_TIEMPOS", "salida/comparacion_modelos/tiempos_asignacion.csv"
    ))

    with open(entrada, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    if subset_path:
        urls_subset = leer_urls_subset(Path(subset_path))
        filas = [f for f in filas if f["url"] in urls_subset]
        print(f"[{slug}] acotado a {len(filas)} URLs vía {subset_path}", flush=True)

    print(f"[{slug}] {len(filas)} URLs leídas de {entrada}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(modelo)
    max_tokens_prompt = max_model_len - 300 - MARGEN_SEGURIDAD_TOKENS

    prompts = []
    indices = []  # fila_idx por cada prompt, mismo orden que prompts
    recortados_por_token = 0
    for fila_idx, fila in enumerate(filas):
        texto = (fila.get("comentarios_licencia") or "").strip()
        if not texto:
            continue
        prompt = PROMPT_ASIGNAR.format(
            diccionario_licencias=DICCIONARIO_LICENCIAS_TXT,
            texto=texto[:MAX_CHARS_LICENCIA],
        )
        prompt_recortado = recortar_a_tokens(tokenizer, prompt, max_tokens_prompt)
        if prompt_recortado is not prompt:
            recortados_por_token += 1
        prompts.append(prompt_recortado)
        indices.append(fila_idx)

    print(
        f"[{slug}] {len(prompts)} URLs con texto a asignar licencia "
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

    sampling = SamplingParams(temperature=temperatura, max_tokens=300)

    # chat() en vez de generate(): aplica el chat template del modelo --
    # ver docstring del módulo (esto es justo lo que causaba el 56% de
    # "(sin match)" con Qwen3-Next-80B-A3B, confirmado 2026-10-05).
    conversaciones = [[{"role": "user", "content": p}] for p in prompts]

    t_asignar_inicio = time.time()
    salidas = llm.chat(conversaciones, sampling) if conversaciones else []
    t_asignar = time.time() - t_asignar_inicio

    asignaciones: dict[int, tuple[str, str]] = {
        fila_idx: parsear_asignacion(salida_llm.outputs[0].text)
        for fila_idx, salida_llm in zip(indices, salidas)
    }

    conteo_licencia: dict[str, int] = {}
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_SALIDA)
        escritor.writeheader()
        for fila_idx, fila in enumerate(filas):
            if fila_idx in asignaciones:
                licencia_asignada, justificacion = asignaciones[fila_idx]
            else:
                licencia_asignada, justificacion = "", "Sin texto de licencia/términos/privacidad detectado."
            conteo_licencia[licencia_asignada] = conteo_licencia.get(licencia_asignada, 0) + 1
            escritor.writerow({
                "url": fila["url"],
                "licencia_detectada": fila.get("licencia", ""),
                "licencia_asignada_llm": licencia_asignada,
                "justificacion_asignacion_llm": justificacion,
            })

    t_total = time.time() - t_inicio
    print(f"[{slug}] listo, {len(filas)} URLs procesadas -> {salida}", flush=True)
    print(f"[{slug}] conteo de licencia_asignada_llm: {conteo_licencia}", flush=True)
    print(
        f"[{slug}] tiempos -- carga modelo: {t_carga:.1f}s, asignar: {t_asignar:.1f}s, "
        f"total: {t_total:.1f}s",
        flush=True,
    )

    campos_tiempos = [
        "modelo", "temperatura", "gpus", "urls", "urls_con_texto",
        "tiempo_carga_s", "tiempo_asignar_s", "tiempo_total_s", "conteo_licencias",
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
            "urls_con_texto": len(prompts),
            "tiempo_carga_s": f"{t_carga:.1f}",
            "tiempo_asignar_s": f"{t_asignar:.1f}",
            "tiempo_total_s": f"{t_total:.1f}",
            "conteo_licencias": json.dumps(conteo_licencia, ensure_ascii=False),
        })
    print(f"[{slug}] tiempos registrados -> {tiempos_csv}", flush=True)


if __name__ == "__main__":
    main()
