"""Evalúa con Qwen2.5-32B-Instruct, vía el servidor vLLM levantado por
slurm/vllm_server_licencias.sbatch, si el contenido de cada URL detectada
en salida/licencias_detectadas.jsonl se puede usar o no.

A diferencia de la versión legada (scraping/extras/licencias/ del repo
padre, antes de que esto se separara en su propio repo), que concatenaba TODO el texto de una URL
(footer + subpágina + PDF...) en un solo bloque y lo mandaba en una sola
consulta -- con el riesgo de que el recorte a MAX_CHARS_LICENCIA se comiera
contenido relevante de un documento posterior -- acá cada "unidad de texto"
(ver `unidades_de_resultado` en 01_detectar_licencia.py: cada pieza de footer,
cada subpágina, cada PDF, por separado) se evalúa con su PROPIA consulta al
modelo, y el veredicto final de la URL sale de combinar los veredictos de
todas sus unidades:

  - si CUALQUIER unidad da NO -> la URL es NO (un solo documento
    restrictivo alcanza para bloquear el uso, aunque otro diga que sí).
  - si no hay ningún NO pero alguna unidad da DEPENDE -> la URL es DEPENDE
    (se le pone además, en la misma pasada, un nivel 0-5 -- ver más abajo).
  - si TODAS las unidades dan SI -> la URL es SI.
  - si no hay ninguna unidad con texto -> SIN_TEXTO, sin llamar al modelo.

El detalle de cada unidad (categoría, vía, veredicto y justificación
propios) queda en la columna `respuesta_cruda_llm` (JSON), para poder
auditar de dónde salió el veredicto combinado sin volver a consultar el
modelo.

Si el veredicto combinado da DEPENDE, en la MISMA pasada (mismo request a
la URL, mismo `detalle` todavía en memoria) se le pide además al modelo un
nivel 0-5 de qué tan cerca está de poder usarse (`nivel_uso_llm`,
`explicacion_nivel_llm`) -- es el paso que antes hacía
`matizar_depende_llm.py` como una corrida aparte, releyendo
`respuesta_cruda_llm` del csv ya escrito. Fusionado acá para poder comparar
evaluador por evaluador (un modelo nuevo = un solo script/corrida, no dos)
sin encadenar dos sbatch distintos por modelo. `matizar_depende_llm.py`
queda solo para backfill de csvs viejos que se generaron antes de esta
fusión (ej. las corridas de producción con Qwen 32B/72B ya existentes).

Reanudable por `url`: al arrancar carga licencias_evaluadas.csv si ya
existe y saltea las urls ya procesadas.

Modo continuo: 01_detectar_licencia.py escribe licencias_detectadas.jsonl
append-only y puede tardar horas/días en terminar. Por eso este script no
hace una sola pasada: después de procesar lo que encuentre, si no hay
filas nuevas espera POLL_SEGUNDOS y vuelve a mirar el jsonl de entrada,
indefinidamente. Corta al llegar al --time del sbatch; como es reanudable,
un `sbatch` de nuevo retoma donde quedó.

Tiene que correr en el mismo nodo que el servidor vLLM (ver
slurm/detectar_licencias_evaluar.sbatch) -- conectar de un nodo a otro está
bloqueado por el firewall del cluster, confirmado 2026-09-17.
"""

import csv
import glob
import json
import re
import time
from pathlib import Path

# respuesta_cruda_llm puede llevar texto SIN recortar (el respaldo de
# compatibilidad de unidades_de_fila lee comentarios_licencia completo, sin
# el tope de 8000 caracteres que aplica 01_detectar_licencia.py a esa columna)
# -- el límite default del módulo csv (128 KB) se queda corto para eso y
# tira "_csv.Error: field larger than field limit" al releer el archivo.
csv.field_size_limit(10_000_000)

import os
import requests

ENTRADA = Path("salida/licencias_detectadas.jsonl")
ENTRADA_CSV = Path("salida/licencias_detectadas.csv")
# EVAL_SALIDA/EVAL_MODELO/EVAL_ENDPOINT_GLOB: overrides por variable de
# entorno para poder correr el MISMO script contra un modelo distinto (ej.
# Qwen2.5-72B-Instruct para comparar tiempos/resultados contra el 32B) sin
# duplicar el archivo -- ver slurm/detectar_licencias_evaluar_qwen72b.sbatch.
# Default: comportamiento normal (Qwen 32B, salida de siempre).
SALIDA = Path(os.environ.get("EVAL_SALIDA", "salida/licencias_evaluadas.csv"))

# Compat: filas del jsonl escritas por una corrida del detector que
# arrancó ANTES de que existiera el campo "unidades" (un proceso ya
# corriendo no relee el .py del disco) no lo tienen -- para esas se cae a
# leer comentarios_licencia del csv como una única unidad "todo junto",
# igual que hacía la versión anterior del evaluador. Ver
# `unidades_de_fila`.
RE_RECORTADO = re.compile(r"\[\.\.\.recortado -- texto completo en: (.+?)\]")
MODELO = os.environ.get(
    "EVAL_MODELO", "/workspace1-ranokau/projects/datatrainlatamgpt/models/Qwen2.5-32B-Instruct"
)
ENDPOINT_GLOB = os.environ.get("EVAL_ENDPOINT_GLOB", "slurm_logs/vllm_endpoint_*.txt")
MAX_CHARS_LICENCIA = 5000  # margen bajo max_model_len=4096 tokens del servidor
# (cota en caracteres, no en tokens -- texto denso puede tener bastante menos
# de los ~3.5 caracteres por token que asume esta cota, así que se dejó
# margen extra -- confirmado con un caso real en la versión anterior). Acá
# aplica POR UNIDAD, no sobre el total concatenado de la URL.
POLL_SEGUNDOS = 30  # espera entre relecturas de ENTRADA cuando no hay filas nuevas

PROMPT = """Eres un asistente legal. Te paso un texto encontrado en el sitio de \
origen de un contenido, relacionado con su uso permitido -- puede ser una \
licencia (ej. Creative Commons, Apache), términos y condiciones de uso, una \
política de privacidad/datos, u otro tipo de texto legal del sitio.
Responde SOLO en este formato:
VEREDICTO: SI | NO | DEPENDE
JUSTIFICACION: <una o dos frases explicando por qué, citando la cláusula relevante>

Texto encontrado:
{licencia}
"""

# Mismo prompt que matizar_depende_llm.py -- duplicado a propósito (no
# importado desde ahí) para no crear un import circular: matizar_depende_llm.py
# ya importa de este módulo (MODELO, SALIDA, endpoint_localhost...).
PROMPT_MATIZ = """Eres un asistente legal. Ya evaluaste este texto legal y contestaste \
DEPENDE sobre si se puede usar el contenido. Ahora necesito que le pongas \
un nivel del 0 al 5 de qué tan cerca está de poder usarse:
0 = definitivamente NO se puede usar
1 = probablemente NO se puede usar
2 = más inclinado a NO, pero no del todo claro
3 = más inclinado a SI, pero no del todo claro
4 = probablemente SI se puede usar
5 = definitivamente SI se puede usar

Responde SOLO en este formato:
NIVEL: <0-5>
EXPLICACION: <una o dos frases explicando por qué ese nivel>

Texto encontrado:
{licencia}

Tu evaluación anterior:
Justificación: {justificacion_llm}
"""

CAMPOS_SALIDA = [
    "url",
    "licencia",
    "veredicto_llm",
    "justificacion_llm",
    "respuesta_cruda_llm",
    "nivel_uso_llm",
    "explicacion_nivel_llm",
]

# Orden de severidad para combinar los veredictos de las unidades de una
# misma URL: el más restrictivo gana -- ver docstring del módulo.
_PRIORIDAD = {"NO": 0, "DEPENDE": 1, "ERROR": 2, "SI": 3}


def endpoint_localhost() -> int:
    # ordenar por fecha de modificación, NO por nombre: alfabético rompe en
    # cuanto un jobid cruza de 4 a 5 dígitos ("10731" < "9817" como string)
    archivos = sorted(glob.glob(ENDPOINT_GLOB), key=lambda p: Path(p).stat().st_mtime)
    if not archivos:
        raise SystemExit(
            f"No hay ningún {ENDPOINT_GLOB} -- ¿está corriendo el servidor vLLM "
            "correspondiente?"
        )
    linea = open(archivos[-1]).read().strip()
    m = re.match(r"host=(\S+) port=(\d+)", linea)
    if not m:
        raise SystemExit(f"No pude parsear {archivos[-1]!r}: {linea!r}")
    return int(m.group(2))


def parsear_respuesta(texto: str) -> tuple[str, str]:
    m_veredicto = re.search(r"VEREDICTO:\s*(SI|NO|DEPENDE)", texto, re.IGNORECASE)
    m_justificacion = re.search(r"JUSTIFICACION:\s*(.+)", texto, re.IGNORECASE | re.DOTALL)
    veredicto = m_veredicto.group(1).upper() if m_veredicto else "DEPENDE"
    justificacion = m_justificacion.group(1).strip() if m_justificacion else texto.strip()
    return veredicto, justificacion


class ErrorLLM(Exception):
    """El servidor vLLM respondió con un error (ej. 400 por exceder
    max_model_len)."""


def preguntar(port: int, texto_unidad: str) -> tuple[str, str]:
    prompt = PROMPT.format(licencia=texto_unidad[:MAX_CHARS_LICENCIA])
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": MODELO,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
        },
        timeout=120,
    )
    if not resp.ok:
        raise ErrorLLM(f"HTTP {resp.status_code}: {resp.text[:500]}")
    respuesta = resp.json()["choices"][0]["message"]["content"]
    return parsear_respuesta(respuesta)


def parsear_nivel(texto: str) -> tuple[str, str]:
    m_n = re.search(r"NIVEL:\s*([0-5])", texto)
    m_e = re.search(r"EXPLICACION:\s*(.+)", texto, re.IGNORECASE | re.DOTALL)
    nivel = m_n.group(1) if m_n else ""
    explicacion = m_e.group(1).strip() if m_e else texto.strip()
    return nivel, explicacion


def preguntar_nivel(port: int, texto_licencia: str, justificacion_llm: str) -> tuple[str, str]:
    prompt = PROMPT_MATIZ.format(
        licencia=texto_licencia[:MAX_CHARS_LICENCIA],
        justificacion_llm=justificacion_llm,
    )
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": MODELO,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
        },
        timeout=120,
    )
    if not resp.ok:
        raise ErrorLLM(f"HTTP {resp.status_code}: {resp.text[:500]}")
    respuesta = resp.json()["choices"][0]["message"]["content"]
    return parsear_nivel(respuesta)


def texto_de_unidad(unidad: dict) -> str:
    """Réplica liviana de texto_de_unidad de 01_detectar_licencia.py -- no se
    importa ese módulo para no arrastrar sus dependencias pesadas
    (requests/bs4/markdownify/pdf_a_texto) al venv aparte de este script."""
    if unidad.get("markdown_path"):
        try:
            return Path(unidad["markdown_path"]).read_text(encoding="utf-8")
        except OSError:
            return ""
    return unidad.get("texto_inline") or ""


def evaluar_unidades(port: int, unidades: list[dict]) -> tuple[str, str, list[dict]]:
    """Evalúa cada unidad por separado y combina los veredictos (el más
    restrictivo gana -- ver _PRIORIDAD). Devuelve (veredicto_combinado,
    justificacion_combinada, detalle) -- detalle es una lista con el
    veredicto/justificación de cada unidad individual, para auditar."""
    detalle = []
    for unidad in unidades:
        texto = texto_de_unidad(unidad).strip()
        if not texto:
            continue
        try:
            veredicto, justificacion = preguntar(port, texto)
        except (ErrorLLM, requests.RequestException) as e:
            veredicto, justificacion = "ERROR", f"Falló la consulta al modelo: {e}"
        detalle.append({
            "categoria": unidad.get("categoria"),
            "via": unidad.get("via"),
            "url_origen": unidad.get("url_origen"),
            # se guarda cómo recuperar el texto de ESTA unidad puntual (no
            # el texto en sí, salvo que sea inline) -- lo usa texto_depende_de
            # más abajo (misma pasada) y, para csvs viejos sin este campo
            # fusionado, matizar_depende_llm.py releyéndolo del csv.
            "markdown_path": unidad.get("markdown_path"),
            "texto_inline": unidad.get("texto_inline"),
            "veredicto": veredicto,
            "justificacion": justificacion,
        })

    if not detalle:
        return "SIN_TEXTO", "No se detectó texto de licencia/términos/privacidad en el sitio.", []

    veredicto_combinado, justificacion_combinada = combinar_detalle(detalle)
    return veredicto_combinado, justificacion_combinada, detalle


def combinar_detalle(detalle: list[dict]) -> tuple[str, str]:
    """El veredicto más restrictivo de `detalle` gana -- ver _PRIORIDAD.
    Extraído de evaluar_unidades para poder reusarlo desde un flujo batch
    (02_evaluar_licencias_llm.py) que arma el `detalle` de otra forma (todos
    los prompts de golpe vía vllm.LLM().generate(), no un request HTTP por
    unidad) pero necesita combinar igual. Asume `detalle` no vacío --
    SIN_TEXTO se decide antes, sin llamar a esto."""
    peor = max(detalle, key=lambda d: -_PRIORIDAD.get(d["veredicto"], 1))
    veredicto_combinado = peor["veredicto"]
    # La justificación combinada junta las de TODAS las unidades que
    # compartan el veredicto ganador (puede haber más de un NO, por
    # ejemplo) -- no solo la primera -- para no perder contexto si dos
    # documentos distintos son restrictivos por motivos distintos.
    coincidentes = [d["justificacion"] for d in detalle if d["veredicto"] == veredicto_combinado]
    justificacion_combinada = " | ".join(coincidentes)
    return veredicto_combinado, justificacion_combinada


def texto_depende_de(detalle: list[dict]) -> str:
    """Junta el texto de las unidades que individualmente dieron DEPENDE
    (no todas las de la URL) -- misma lógica que matizar_depende_llm.py,
    pero sobre el `detalle` que ya está en memoria de esta misma pasada, en
    vez de releerlo desde la columna respuesta_cruda_llm de un csv."""
    partes = [texto_de_unidad(d) for d in detalle if d.get("veredicto") == "DEPENDE"]
    return "\n\n---\n\n".join(p.strip() for p in partes if p and p.strip())


def ya_procesadas() -> set[str]:
    if not SALIDA.exists():
        return set()
    with open(SALIDA, newline="", encoding="utf-8") as f:
        return {fila["url"] for fila in csv.DictReader(f)}


def leer_entrada() -> list[dict]:
    if not ENTRADA.exists():
        return []
    filas = []
    with open(ENTRADA, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                filas.append(json.loads(linea))
    return filas


def _comentarios_por_url_csv() -> dict[str, str]:
    if not ENTRADA_CSV.exists():
        return {}
    with open(ENTRADA_CSV, newline="", encoding="utf-8") as f:
        return {fila["url"]: fila.get("comentarios_licencia", "") for fila in csv.DictReader(f)}


def _texto_comentarios_completo(comentarios: str) -> str:
    """Resuelve el aviso de recorte de comentarios_licencia leyendo los
    .md completos -- mismo mecanismo que usaba la versión anterior del
    evaluador (ver la versión legada en scraping/extras/licencias/ del repo padre)."""
    m = RE_RECORTADO.search(comentarios)
    if not m:
        return comentarios.strip()
    partes = []
    for ruta in (r.strip() for r in m.group(1).split(";")):
        try:
            partes.append(Path(ruta).read_text(encoding="utf-8"))
        except OSError as e:
            print(f"  aviso: no pude leer {ruta!r} ({e}), sigo sin ese archivo", flush=True)
    return "\n\n---\n\n".join(partes).strip() if partes else comentarios.strip()


def unidades_de_fila(fila: dict, comentarios_por_url: dict[str, str]) -> list[dict]:
    """`fila["unidades"]` si el detector que la escribió ya las generaba;
    si no (jsonl viejo, ver nota junto a ENTRADA_CSV arriba), cae a una
    única unidad sintética con comentarios_licencia completo del csv --
    peor que evaluar cada documento por separado, pero mejor que perder el
    texto entero y marcar SIN_TEXTO por error."""
    unidades = fila.get("unidades")
    if unidades:
        return unidades
    comentarios = comentarios_por_url.get(fila["url"], "")
    if not comentarios:
        return []
    texto = _texto_comentarios_completo(comentarios)
    if not texto:
        return []
    return [{
        "categoria": None, "via": fila.get("via"), "url_origen": fila["url"],
        "markdown_path": None, "texto_inline": texto,
    }]


def main() -> None:
    port = endpoint_localhost()
    procesadas = ya_procesadas()
    es_nuevo = not SALIDA.exists()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)  # SALIDA vive en una carpeta por modelo

    with open(SALIDA, "a", newline="", encoding="utf-8") as f_out:
        escritor = csv.DictWriter(f_out, fieldnames=CAMPOS_SALIDA)
        if es_nuevo:
            escritor.writeheader()
            f_out.flush()

        total = 0
        while True:
            nuevas_esta_vuelta = 0
            filas = leer_entrada()
            comentarios_por_url = _comentarios_por_url_csv()
            for i, fila in enumerate(filas, start=1):
                url = fila["url"]
                if url in procesadas:
                    continue

                veredicto, justificacion, detalle = evaluar_unidades(
                    port, unidades_de_fila(fila, comentarios_por_url)
                )

                nivel, explicacion_nivel = "", ""
                if veredicto == "DEPENDE":
                    texto_depende = texto_depende_de(detalle)
                    if texto_depende:
                        try:
                            nivel, explicacion_nivel = preguntar_nivel(port, texto_depende, justificacion)
                        except (ErrorLLM, requests.RequestException) as e:
                            explicacion_nivel = f"Error al matizar: {e}"
                    else:
                        # No debería pasar (DEPENDE implica al menos una
                        # unidad con ese veredicto), pero se deja la guarda
                        # igual que hacía matizar_depende_llm.py.
                        explicacion_nivel = "Sin texto de licencia para matizar."

                escritor.writerow({
                    "url": url,
                    "licencia": fila.get("licencia", ""),
                    "veredicto_llm": veredicto,
                    "justificacion_llm": justificacion,
                    "respuesta_cruda_llm": json.dumps(detalle, ensure_ascii=False),
                    "nivel_uso_llm": nivel,
                    "explicacion_nivel_llm": explicacion_nivel,
                })
                f_out.flush()
                procesadas.add(url)
                total += 1
                nuevas_esta_vuelta += 1
                sufijo_nivel = f", nivel {nivel or '?'}" if veredicto == "DEPENDE" else ""
                print(f"[{i}] {url} -> {veredicto}{sufijo_nivel} ({len(detalle)} unidad(es))", flush=True)

            if nuevas_esta_vuelta == 0:
                print(
                    f"sin filas nuevas ({total} procesadas en total), "
                    f"esperando {POLL_SEGUNDOS}s...",
                    flush=True,
                )
                time.sleep(POLL_SEGUNDOS)


if __name__ == "__main__":
    main()
