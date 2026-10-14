"""Juez: usa Llama-3.3-70B-Instruct con vLLM en modo offline/batch (una sola
corrida, no levanta servidor, 2 GPUs) para comparar los veredictos que dio
un evaluador (salida/licencias_evaluadas.csv, o el
archivo que se le indique vía JUZGAR_EVALUADAS -- ver más abajo) contra la
columna "Comentarios Licencia" de la hoja de referencia manual
(licencias_original_a_mano.xlsx, primera hoja --
hoy "Descargados Lascar", pero se toma por posición, no por nombre), AHORA
TAMBIÉN con el diccionario de licencias (dicc_lic.json --
20 licencias conocidas con su descripción en términos de qué permiten/
restringen, transcripto de dicc_lic.png) como referencia extra para que el
juez pueda chequear si el veredicto del modelo es consistente con lo que esa
licencia realmente permite, no solo si "suena parecido" al comentario
manual.

Adaptado de la versión legada en scraping/extras/licencias/ del repo padre -- misma
lógica y mismo motivo para ignorar la columna "Licencia" (categoría corta,
no comparable directo contra un veredicto SI/NO/DEPENDE; ver ese script
para el detalle). Dos diferencias:

  1. El veredicto del juez es BIEN | MAL | PARCIAL (pasó por MAS_O_MENOS en
     el medio, vuelto a PARCIAL -- mismo concepto, a pedido del usuario).
  2. Las filas cuya fila de referencia menciona "alianza" (ej. "Alianza
     Wikimedia" -- convenios/colaboraciones, no un comentario real sobre
     los términos de uso del sitio) se EXCLUYEN de juzgar -- no hay nada
     legal que comparar ahí -- pero se cuentan aparte para el análisis
     final, no se descartan en silencio.

Se omiten además, igual que la versión anterior: filas con
veredicto_llm == "SIN_TEXTO" (el detector no encontró nada, no hay nada
que juzgar -- reemplaza el chequeo de `via` vacía de la versión anterior,
ya que SIN_TEXTO es ahora su propio veredicto en vez de un DEPENDE por
default) y filas sin fila correspondiente en la hoja de referencia.

Variables de entorno (mismo patrón que compartido/evaluar_licencias_modo_server.py, para
poder juzgar el csv de Qwen 32B y el de Qwen 72B con el mismo script sin
duplicar código):
  JUZGAR_EVALUADAS: csv a juzgar (default: licencias_evaluadas.csv)
  JUZGAR_SALIDA: csv de salida (default: juicio_llama.csv)

A diferencia de compartido/evaluar_licencias_modo_server.py, esto NO es continuo/reanudable:
es una sola pasada batch sobre datos ya fijos (no un csv que sigue
creciendo), así que carga el modelo, procesa todo de una vez y termina.

El diccionario de licencias agrega ~2575 caracteres fijos a CADA prompt
(siempre el mismo bloque, ~700-900 tokens estimados) -- por eso
`max_model_len` subió de 2048 a 4096 acá (antes alcanzaba con 2048 porque
el prompt solo llevaba comentario manual + veredicto + justificación, sin
este bloque fijo). Si tira OOM al cargar Llama-3.3-70B en 2 GPUs con este
max_model_len más grande, los ajustes a probar en orden: (1)
`--enforce-eager` (se salta CUDA graphs, usa menos memoria a cambio de algo
de velocidad -- mismo truco que ya documenta vllm_server_llama_juez.sbatch),
(2) subir a 3-4 GPUs.

Usa `llm.chat()` (no `llm.generate()`): chat() aplica el chat template del
modelo (marcas de turno user/assistant), generate() manda el prompt como
texto plano -- sin esas marcas el modelo puede terminar CONTINUANDO el
texto en vez de responder como asistente (confirmado con Qwen3-Next-80B-A3B
en 04_asignar_licencia_llm.py, 2026-10-05: 56% de las respuestas no seguían el
formato pedido). También tiene el mismo recorte de respaldo por tokens
reales que 02_evaluar_licencias_llm.py (recortar_a_tokens) -- un solo
prompt pasado de largo tumba TODO el batch en modo offline.
"""

import csv
import importlib
import json
import os
import re
from pathlib import Path

import openpyxl
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# import normal no funciona: el nombre del módulo empieza con dígito
# (prefijo numérico del pipeline) -- ver 02_evaluar_licencias_llm.py.
recortar_a_tokens = importlib.import_module("02_evaluar_licencias_llm").recortar_a_tokens

# licencias_evaluadas.csv puede tener columnas (respuesta_cruda_llm) con
# texto sin recortar que supera el límite default del módulo csv (128 KB)
# -- ver el mismo fix en compartido/evaluar_licencias_modo_server.py.
csv.field_size_limit(10_000_000)

MODELO = "/workspace1-ranokau/projects/datatrainlatamgpt/models/Llama-3.3-70B-Instruct"
MAX_MODEL_LEN = 4096
MARGEN_SEGURIDAD_TOKENS = 100  # colchón para el chat template + salida del juez
EVALUADAS = Path(os.environ.get("JUZGAR_EVALUADAS", "salida/licencias_evaluadas.csv"))
REFERENCIA = Path("referencia/licencias_original_a_mano.xlsx")
DICC_LICENCIAS = Path("referencia/dicc_lic.json")
SALIDA = Path(os.environ.get("JUZGAR_SALIDA", "salida/juicio_llama.csv"))


def _cargar_diccionario_licencias() -> str:
    """Arma el bloque de texto fijo con las licencias conocidas (nombre +
    descripción, sin fuente_principal -- no aporta nada al juicio, solo
    gastaría tokens) para insertar en PROMPT_JUEZ."""
    with open(DICC_LICENCIAS, encoding="utf-8") as f:
        dicc = json.load(f)
    return "\n".join(f"- {lic}: {info['descripcion']}" for lic, info in dicc.items())


DICCIONARIO_LICENCIAS_TXT = _cargar_diccionario_licencias()

PROMPT_JUEZ = """Eres un revisor legal. Te paso dos evaluaciones sobre si se puede \
usar el contenido de una fuente, según su licencia: una clasificación hecha \
a mano por un humano (referencia), y un veredicto que dio otro modelo de \
lenguaje (a evaluar).

Además tenés un diccionario de licencias conocidas, con su descripción en \
términos de qué permiten y qué restringen -- usalo como referencia cuando \
el comentario manual o la justificación mencionen alguna de estas licencias, \
para confirmar si el veredicto del modelo (SI/NO/DEPENDE) es consistente \
con lo que esa licencia realmente permite.

Diccionario de licencias:
{diccionario_licencias}

Tu tarea es juzgar si el veredicto del modelo coincide con la clasificación \
manual (y, cuando aplique, con lo que indica el diccionario de licencias de \
arriba).
Responde SOLO en este formato:
VEREDICTO: BIEN | MAL | PARCIAL
EXPLICACION: <una o dos frases explicando por qué>

Comentarios de licencia (referencia, escritos a mano):
{comentarios_manual}

Veredicto del modelo a evaluar:
Veredicto: {veredicto_llm}
Justificación: {justificacion_llm}
"""

CAMPOS_SALIDA = [
    "url",
    "licencia_manual",
    "comentarios_licencia_manual",
    "veredicto_llm",
    "justificacion_llm",
    "veredicto_juez",
    "explicacion_juez",
]


def cargar_referencia() -> dict[str, tuple[str, str]]:
    wb = openpyxl.load_workbook(REFERENCIA, read_only=True)
    ws = wb.worksheets[0]  # solo la primera hoja (hoy: "Descargados Lascar")
    filas = ws.iter_rows(values_only=True)
    encabezado = list(next(filas))
    idx_url = encabezado.index("URL/Alianza")
    idx_licencia = encabezado.index("Licencia")
    idx_comentarios = encabezado.index("Comentarios Licencia")

    # Algunas URLs aparecen en más de una fila (mismo origen, distintos
    # sub-datasets), a veces con Licencia/Comentarios distintos entre sí
    # (ej. RedPajama-Data aparece 3 veces) -- se combinan todos los valores
    # únicos en vez de quedarse con el último en silencio.
    por_url: dict[str, list[tuple[str, str]]] = {}
    for fila in filas:
        url = fila[idx_url]
        if not url:
            continue
        url = str(url).strip()
        licencia = str(fila[idx_licencia] or "").strip()
        comentarios = str(fila[idx_comentarios] or "").strip()
        por_url.setdefault(url, []).append((licencia, comentarios))

    referencia = {}
    for url, valores in por_url.items():
        licencias = list(dict.fromkeys(v[0] for v in valores if v[0]))
        comentarios = list(dict.fromkeys(v[1] for v in valores if v[1]))
        referencia[url] = ("; ".join(licencias), "\n\n---\n\n".join(comentarios))
    return referencia


def parsear_juicio(texto: str) -> tuple[str, str]:
    m_v = re.search(r"VEREDICTO:\s*(BIEN|MAL|PARCIAL)", texto, re.IGNORECASE)
    m_e = re.search(r"EXPLICACION:\s*(.+)", texto, re.IGNORECASE | re.DOTALL)
    veredicto = m_v.group(1).upper() if m_v else "PARCIAL"
    explicacion = m_e.group(1).strip() if m_e else texto.strip()
    return veredicto, explicacion


def main() -> None:
    referencia = cargar_referencia()

    with open(EVALUADAS, newline="", encoding="utf-8") as f:
        evaluadas = list(csv.DictReader(f))

    filas_a_juzgar = []
    omitidas_sin_texto = 0
    omitidas_sin_referencia = 0
    omitidas_alianza = 0
    for fila in evaluadas:
        url = fila["url"]
        if fila["veredicto_llm"] == "SIN_TEXTO":
            omitidas_sin_texto += 1
            continue
        if url not in referencia:
            omitidas_sin_referencia += 1
            continue
        licencia_manual, comentarios_manual = referencia[url]
        if "alianza" in licencia_manual.lower() or "alianza" in comentarios_manual.lower():
            omitidas_alianza += 1
            continue
        filas_a_juzgar.append(fila)

    print(
        f"{len(filas_a_juzgar)} filas para juzgar de {len(evaluadas)} evaluadas "
        f"({omitidas_sin_texto} sin texto detectado, "
        f"{omitidas_sin_referencia} sin fila de referencia manual, "
        f"{omitidas_alianza} excluidas por mencionar 'alianza')",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    max_tokens_prompt = MAX_MODEL_LEN - 300 - MARGEN_SEGURIDAD_TOKENS

    prompts = []
    recortados_por_token = 0
    for fila in filas_a_juzgar:
        prompt = PROMPT_JUEZ.format(
            diccionario_licencias=DICCIONARIO_LICENCIAS_TXT,
            comentarios_manual=referencia[fila["url"]][1] or "(vacío)",
            veredicto_llm=fila["veredicto_llm"],
            justificacion_llm=fila["justificacion_llm"],
        )
        prompt_recortado = recortar_a_tokens(tokenizer, prompt, max_tokens_prompt)
        if prompt_recortado is not prompt:
            recortados_por_token += 1
        prompts.append(prompt_recortado)
    print(f"{recortados_por_token} prompts recortados por exceder tokens del modelo", flush=True)

    llm = LLM(
        model=MODELO,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=MAX_MODEL_LEN,
    )
    sampling = SamplingParams(temperature=0, max_tokens=300)
    # chat() en vez de generate(): aplica el chat template del modelo -- ver
    # docstring del módulo.
    conversaciones = [[{"role": "user", "content": p}] for p in prompts]
    salidas = llm.chat(conversaciones, sampling)

    conteo_juicio: dict[str, int] = {}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)  # SALIDA vive en una carpeta por modelo
    with open(SALIDA, "w", newline="", encoding="utf-8") as f_out:
        escritor = csv.DictWriter(f_out, fieldnames=CAMPOS_SALIDA)
        escritor.writeheader()
        for fila, salida in zip(filas_a_juzgar, salidas):
            licencia_manual, comentarios_manual = referencia[fila["url"]]
            texto = salida.outputs[0].text
            veredicto_juez, explicacion_juez = parsear_juicio(texto)
            conteo_juicio[veredicto_juez] = conteo_juicio.get(veredicto_juez, 0) + 1
            escritor.writerow(
                {
                    "url": fila["url"],
                    "licencia_manual": licencia_manual,
                    "comentarios_licencia_manual": comentarios_manual,
                    "veredicto_llm": fila["veredicto_llm"],
                    "justificacion_llm": fila["justificacion_llm"],
                    "veredicto_juez": veredicto_juez,
                    "explicacion_juez": explicacion_juez,
                }
            )

    print(f"listo, {len(filas_a_juzgar)} filas juzgadas -> {SALIDA}", flush=True)
    print(f"conteo de veredicto_juez: {conteo_juicio}", flush=True)
    print(
        f"análisis final -- evaluadas: {len(evaluadas)}, juzgadas: {len(filas_a_juzgar)}, "
        f"sin texto: {omitidas_sin_texto}, sin referencia: {omitidas_sin_referencia}, "
        f"excluidas por alianza: {omitidas_alianza}",
        flush=True,
    )


if __name__ == "__main__":
    main()
