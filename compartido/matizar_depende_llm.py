"""BACKFILL -- para csvs generados ANTES de que evaluar_licencias_modo_server.py
matizara en la misma pasada (ver su docstring). Para cualquier corrida
nueva (incluida la comparación entre modelos Qwen), evaluar_licencias_modo_server.py
ya escribe nivel_uso_llm/explicacion_nivel_llm directo, así que este script
no hace falta correrlo aparte -- se mantiene solo para completar
licencias_evaluadas.csv/licencias_evaluadas_qwen72b.csv, generados con la
versión vieja de dos pasadas.

Matiza el veredicto DEPENDE de evaluar_licencias_modo_server.py: para cada fila
con `veredicto_llm == "DEPENDE"`, le pide a Qwen2.5-32B-Instruct -- mismo
servidor vLLM, vía localhost -- un nivel del 0 al 5 de qué tan cerca está
de poder usarse el contenido, con explicación.

A diferencia de la versión anterior, que le pasaba al modelo TODO el texto
de la URL, acá se usa solo el texto de la(s) unidad(es) que individualmente
dieron DEPENDE (extraídas de `respuesta_cruda_llm`, que ya trae cómo
recuperar el texto de cada una -- ver evaluar_licencias_modo_server.py) -- si una
URL tiene 3 documentos y solo uno es ambiguo, no tiene sentido matizar con
el ruido de los otros dos.

Escala (0-5, sin punto medio -- fuerza a inclinarse hacia NO o hacia SI):
  0 = definitivamente NO se puede usar
  1 = probablemente NO se puede usar
  2 = más inclinado a NO, pero no del todo claro
  3 = más inclinado a SI, pero no del todo claro
  4 = probablemente SI se puede usar
  5 = definitivamente SI se puede usar

A diferencia de evaluar_licencias_modo_server.py (append-only), este script
REESCRIBE salida/licencias_evaluadas.csv agregando dos
columnas nuevas (`nivel_uso_llm`, `explicacion_nivel_llm`) -- vacías para
las filas que no son DEPENDE. Reanudable: guarda cada 20 filas, y una fila
que ya tenga `nivel_uso_llm` cargado no se vuelve a consultar.

Tiene que correr en el mismo nodo que el servidor vLLM de Qwen (ver
slurm/detectar_licencias_matizar.sbatch) -- mismo motivo que
evaluar_licencias_modo_server.py.
"""

import csv
import importlib
import json
import re

import requests

# Backfill del evaluador en modo SERVIDOR específicamente (necesita
# endpoint_localhost/MODELO/SALIDA del server HTTP, no del batch) -- ver
# evaluar_licencias_modo_server.py. import normal no funciona para los
# módulos con prefijo numérico (ver 02_evaluar_licencias_llm.py para el
# detalle del truco); este módulo en particular no lleva número, pero se
# usa importlib igual por consistencia con el resto del pipeline.
_ev = importlib.import_module("evaluar_licencias_modo_server")
MAX_CHARS_LICENCIA = _ev.MAX_CHARS_LICENCIA
MODELO = _ev.MODELO
ARCHIVO = _ev.SALIDA  # mismo archivo que escribe evaluar_licencias_modo_server.py
endpoint_localhost = _ev.endpoint_localhost
texto_de_unidad = _ev.texto_de_unidad

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
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    respuesta = resp.json()["choices"][0]["message"]["content"]
    return parsear_nivel(respuesta)


def texto_depende_de(fila: dict) -> str:
    """Junta el texto de las unidades que individualmente dieron DEPENDE
    (no todas las de la URL) -- lee `respuesta_cruda_llm`, que ya trae
    markdown_path/texto_inline de cada una (ver evaluar_licencias_modo_server.py)."""
    try:
        detalle = json.loads(fila.get("respuesta_cruda_llm") or "[]")
    except json.JSONDecodeError:
        return ""
    partes = [
        texto_de_unidad(d) for d in detalle if d.get("veredicto") == "DEPENDE"
    ]
    return "\n\n---\n\n".join(p.strip() for p in partes if p and p.strip())


def guardar(filas: list[dict]) -> None:
    with open(ARCHIVO, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_SALIDA)
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> None:
    port = endpoint_localhost()

    with open(ARCHIVO, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    pendientes = [
        fila
        for fila in filas
        if fila["veredicto_llm"] == "DEPENDE" and not fila.get("nivel_uso_llm")
    ]
    print(f"{len(pendientes)} filas DEPENDE para matizar", flush=True)

    procesadas = 0
    for fila in pendientes:
        texto_licencia = texto_depende_de(fila)

        if not texto_licencia:
            # No debería pasar (DEPENDE implica al menos una unidad
            # evaluada como tal), pero se deja la guarda por si el detalle
            # viene incompleto de una corrida vieja.
            fila["nivel_uso_llm"] = ""
            fila["explicacion_nivel_llm"] = "Sin texto de licencia para matizar."
            continue

        try:
            nivel, explicacion = preguntar_nivel(port, texto_licencia, fila["justificacion_llm"])
        except (requests.RequestException, RuntimeError) as e:
            print(f"  ERROR matizando {fila['url']}: {e}", flush=True)
            nivel, explicacion = "", f"Error al matizar: {e}"

        fila["nivel_uso_llm"] = nivel
        fila["explicacion_nivel_llm"] = explicacion
        procesadas += 1
        print(f"[{procesadas}/{len(pendientes)}] {fila['url']} -> nivel {nivel or '?'}", flush=True)

        if procesadas % 20 == 0:
            guardar(filas)

    guardar(filas)
    print(f"listo, {procesadas} filas matizadas -> {ARCHIVO}", flush=True)


if __name__ == "__main__":
    main()
