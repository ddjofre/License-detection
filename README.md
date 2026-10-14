# License-detection

Pipeline para detectar, evaluar y clasificar la licencia de uso de datasets/sitios
de LatAm (¿se puede usar este contenido para entrenar un modelo? ¿bajo qué
licencia?), usando LLMs (familia Qwen para evaluar, Llama-3.3-70B como juez).

Nace de `detectar_licencias/` dentro de `webscraping-automatizado` -- acá vive
separado, con su propio repo, para poder iterar y versionar sin arrastrar el
resto de ese monorepo.

## Qué hace el pipeline, de punta a punta

```
                          ┌─────────────────────────────┐
                          │ 01_detectar_licencia.py      │  (CLI, un input .csv/.xlsx/.md)
                          │ rastrea el sitio de origen de │
                          │ cada URL y encuentra su texto │
                          │ legal (licencia/ToS/privacidad)│
                          └──────────────┬───────────────┘
                                         │  salida/licencias_detectadas.csv + .jsonl
                    ┌────────────────────┼────────────────────┐
                    │                    │                     │
                    ▼                    ▼                     ▼
   ┌─────────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
   │ compartido/               │ │ 02_evaluar_licencias_ │ │ 04_asignar_licencia_  │
   │ evaluar_licencias_        │ │ llm.py                │ │ llm.py                │
   │ modo_server.py            │ │ (BATCH: compara varios│ │ asigna la licencia    │
   │ (PRODUCCIÓN: server vLLM  │ │  modelos candidatos,  │ │ del diccionario que   │
   │  + HTTP, corre indefinido,│ │  un modelo por corrida,│ │ mejor corresponda al  │
   │  reanudable)               │ │  sin servidor)         │ │ texto encontrado      │
   └──────────────┬────────────┘ └───────────┬───────────┘ └───────────────────────┘
                  │                           │
                  ▼                           ▼
   ┌─────────────────────────┐   salida/<modelo>/licencias_evaluadas_<modelo>.csv
   │ compartido/               │                │
   │ matizar_depende_llm.py    │                ▼
   │ (BACKFILL -- solo csvs    │   ┌──────────────────────┐
   │  viejos sin matizar       │   │ 03_juzgar_licencias_  │
   │  inline, no hace falta    │   │ llm.py                 │
   │  para corridas nuevas)    │   │ compara el veredicto   │
   └───────────────────────────┘   │ contra la referencia   │
                                    │ manual (BIEN/MAL/      │
                                    │ PARCIAL)                │
                                    └────────────────────────┘
```

**Importante**: `02_evaluar_licencias_llm.py` (el batch) y
`compartido/evaluar_licencias_modo_server.py` (el server) **no son pasos
consecutivos** -- son dos formas alternativas de hacer lo mismo (evaluar
SI/NO/DEPENDE). Se usa una u otra según el caso:
- **Server** (`compartido/evaluar_licencias_modo_server.py`): para la corrida real
  de producción, procesando todo el corpus mientras `01_detectar_licencia.py`
  sigue corriendo. Necesita un servidor vLLM levantado aparte
  (`slurm/sbatch/vllm_server_*.sbatch` -- no incluidos en este repo todavía).
- **Batch** (`02_evaluar_licencias_llm.py`): para comparar varios modelos
  candidatos entre sí sobre un subconjunto fijo y chico de URLs, sin pagar el
  costo de un servidor. Es el que se usó para la comparación documentada en
  `docs/RESULTADOS_comparacion_modelos.md`.

`04_asignar_licencia_llm.py` es independiente de los otros 3 -- solo necesita
la salida de `01_detectar_licencia.py`, no depende de haber evaluado ni
juzgado nada antes.

## Los scripts, uno por uno

### `01_detectar_licencia.py`
CLI standalone (no usa variables de entorno). Dado un archivo de entrada con
URLs, para cada una prueba en cascada: adaptador de plataforma conocida
(HuggingFace/GitHub/DSpace/CKAN) → `.well-known/` → `rel="license"`/JSON-LD →
footer → headings del body → crawl de un nivel → página "Acerca de". Baja el
texto encontrado (HTML→Markdown, PDF→Markdown con OCR si hace falta) a
`salida/archivos/`.

```bash
python 01_detectar_licencia.py entrada.xlsx --salida salida/licencias_detectadas.csv
```

| Flag | Qué hace |
|---|---|
| `entrada` (posicional) | `.csv`, `.xlsx`/`.xls` o `.md` con las URLs |
| `--salida` | archivo de salida (default: `<entrada>_licencias.csv`) |
| `--columna-url` | columna con la URL (default: autodetecta) |
| `--descargas` | carpeta para los archivos bajados (default: `archivos/` junto a `--salida`) |
| `--sin-descargas` | solo detecta la referencia, no baja el contenido completo |

**Salida**: `salida/licencias_detectadas.csv` (columnas `url, licencia,
comentarios_licencia, via, url_referencia`) + `.jsonl` paralelo (mismo
contenido + metadata de archivos bajados, usado por los scripts siguientes) +
`salida/archivos/<slug-de-la-url>/...` con el texto completo bajado.

### `compartido/evaluar_licencias_modo_server.py` -- evaluador de PRODUCCIÓN
Le pega por HTTP a un servidor vLLM ya levantado (`/v1/chat/completions`).
Corre **indefinido**: si `licencias_detectadas.jsonl` no tiene filas nuevas,
espera 30s y vuelve a mirar -- pensado para dejarlo corriendo mientras el
detector sigue trabajando. Reanudable (si corta, retoma leyendo lo ya
escrito). Evalúa SI/NO/DEPENDE por unidad de texto, combina el veredicto de
la URL (el más restrictivo gana) y matiza los DEPENDE en la misma pasada.

| Variable | Default |
|---|---|
| `EVAL_MODELO` | `Qwen2.5-32B-Instruct` |
| `EVAL_SALIDA` | `salida/licencias_evaluadas.csv` |
| `EVAL_ENDPOINT_GLOB` | `slurm_logs/vllm_endpoint_*.txt` (dónde buscar host/puerto del servidor) |

**Salida**: csv con `url, licencia, veredicto_llm, justificacion_llm,
respuesta_cruda_llm, nivel_uso_llm, explicacion_nivel_llm`.

### `02_evaluar_licencias_llm.py` -- comparación de modelos (BATCH)
Carga el modelo él mismo offline (vLLM, sin servidor) y lo compara contra
otros candidatos. Dos tandas de `llm.chat()` sobre el mismo modelo ya
cargado: evaluar (un prompt por unidad) y matizar (un prompt por URL
DEPENDE). Un modelo por corrida -- para comparar varios, se corre varias
veces con distinto `CMP_MODELO` (ver `slurm/sbatch/`). No reanudable, una
sola pasada sobre un subconjunto fijo.

| Variable | Default |
|---|---|
| `CMP_MODELO` | **obligatoria** |
| `CMP_ENTRADA` | el jsonl completo (recomendado: un subconjunto fijo aparte) |
| `CMP_SALIDA` | `salida/<slug del modelo>/licencias_evaluadas_<slug>.csv` |
| `CMP_TENSOR_PARALLEL_SIZE` | `1` (GPUs) |
| `CMP_GPU_MEMORY_UTILIZATION` | `0.9` |
| `CMP_MAX_MODEL_LEN` | `4096` |
| `CMP_TEMPERATURE` | `0` (si es ≠0, el archivo de salida agrega `_tempX`) |
| `CMP_TIEMPOS` | `salida/comparacion_modelos/tiempos.csv` (consolidado, una fila por corrida) |

**Salida**: mismas columnas que el evaluador de producción, más una fila
agregada en `CMP_TIEMPOS` con tiempos de carga/evaluar/matizar y conteo de
veredictos -- ahí se compara entre modelos.

### `compartido/matizar_depende_llm.py` -- BACKFILL, no es un paso activo
Solo hace falta para csvs generados por una versión vieja del evaluador que
no matizaba en la misma pasada. Para corridas nuevas (con cualquiera de los
dos evaluadores de arriba) no hace falta correrlo. Reescribe el csv agregando
`nivel_uso_llm`/`explicacion_nivel_llm` a las filas DEPENDE que no lo tengan.

```bash
python compartido/matizar_depende_llm.py   # desde la raíz del repo, no desde compartido/
```

### `03_juzgar_licencias_llm.py` -- juez
Offline/batch con Llama-3.3-70B-Instruct. Compara el veredicto de un
evaluador (de cualquiera de los dos de arriba) contra la columna "Comentarios
Licencia" de `referencia/licencias_original_a_mano.xlsx` (primera hoja), con
el diccionario de licencias (`referencia/dicc_lic.json`) como referencia
extra. Excluye filas `SIN_TEXTO` y filas cuya referencia menciona "alianza"
(convenios, no términos de uso reales).

| Variable | Default |
|---|---|
| `JUZGAR_EVALUADAS` | `salida/licencias_evaluadas.csv` |
| `JUZGAR_SALIDA` | `salida/juicio_llama.csv` |

**Salida**: `url, licencia_manual, comentarios_licencia_manual, veredicto_llm,
justificacion_llm, veredicto_juez, explicacion_juez` -- veredicto_juez es
`BIEN | MAL | PARCIAL`.

### `04_asignar_licencia_llm.py` -- asignación de licencia
Offline/batch, un prompt por URL (no por unidad -- asignar licencia es una
lectura holística del texto completo, no algo donde "la unidad más
restrictiva gana" tenga sentido). Le pide al modelo elegir, de las 20
licencias de `referencia/dicc_lic.json`, la que mejor corresponda al texto
encontrado, o "Ninguna coincide".

| Variable | Default |
|---|---|
| `ASIG_MODELO` | **obligatoria** |
| `ASIG_ENTRADA` | `salida/licencias_detectadas.csv` |
| `ASIG_SUBSET` | sin acotar (opcional: csv/jsonl con columna `url` para acotar) |
| `ASIG_SALIDA` | `salida/<slug del modelo>/licencia_asignada_<slug>.csv` |
| `ASIG_TENSOR_PARALLEL_SIZE` / `ASIG_GPU_MEMORY_UTILIZATION` / `ASIG_MAX_MODEL_LEN` | `1` / `0.9` / `4096` |
| `ASIG_TEMPERATURE` | `0` |
| `ASIG_TIEMPOS` | `salida/comparacion_modelos/tiempos_asignacion.csv` |

**Salida**: `url, licencia_detectada, licencia_asignada_llm,
justificacion_asignacion_llm`.

## Carpetas

| Carpeta | Qué tiene |
|---|---|
| `compartido/` | Los 2 scripts que no son pasos numerados del pipeline: el evaluador de producción (server) y el backfill de matizado. Paquete Python real (`__init__.py`) -- ver nota de imports más abajo. |
| `docs/` | `PLAN_comparacion_modelos.md` (seguimiento de la comparación de modelos, con checkboxes), `RESULTADOS_comparacion_modelos.md` (resultados completos: qué modelo ganó y por qué, incluye el bug del chat template), `PROMPTS.md` (los 4 prompts exactos que se usan). |
| `referencia/` | `dicc_lic.{png,csv,json}` (20 licencias conocidas con su descripción, usado por el juez y por asignar), `licencias_original_a_mano.xlsx` (clasificación manual de referencia, usado por el juez), `excepciones_latam.{csv,json}` (excepciones legales y licencia por defecto por país de LatAm -- referencia, **todavía no está conectado a ningún script del pipeline**). |
| `salida/` | Todo lo que generan los scripts al correr -- gitignorado el contenido pesado salvo lo que ya esté commiteado. Una carpeta por modelo evaluador (`salida/<slug>/`) + `salida/comparacion_modelos/` (csvs consolidados de tiempos) + `salida/archivos/` (texto bajado por el detector). |
| `slurm/sbatch/` | Los `.sbatch` para correr la comparación de modelos en el cluster (Slurm, partición `lascar`, cuenta `latamgpt`). 24 archivos: 4 modelos × (evaluar + juzgar + asignar) × 2 temperaturas (0 y 0.7). |
| `slurm/logs/` | Donde caen los `.out`/`.err` de cada job (gitignorado). |

## Cómo correr todo

**Entorno**: necesita el venv compartido del cluster con `vllm`,
`transformers`, `openpyxl` instalados (`/workspace1-ranokau/users/ddjofre/venv_licencias`
-- no está incluido en este repo, es compartido con otros proyectos del
cluster). No instalar nada con el Python del sistema.

**Local / interactivo** (sin GPU, para probar lectura de datos o lógica que
no cargue un modelo):
```bash
cd License-detection
/workspace1-ranokau/users/ddjofre/venv_licencias/bin/python3 03_juzgar_licencias_llm.py
```

**Cluster (con GPU)**: todo lo que carga un modelo tiene que ir por Slurm, no
correrse directo en el nodo de login.
```bash
sbatch slurm/sbatch/comparar_evaluadores_qwen2.5-32b.sbatch
squeue --me
tail -f slurm/logs/cmp1_<jobid>.out
```
Cada `.sbatch` fija `REPO=`, activa el venv por ruta completa (no
`source activate`, tiene un bug conocido en el nodo `lascar`) y exporta las
variables de entorno del script correspondiente. Nombres de job cortos
(`cmp1`-`cmp4`, `t1`-`t4`, `job1`-`job4`/`jt1`-`jt4`, `asig1`-`asig4`/`at1`-`at4`)
-- el mapeo completo está en el primer comentario de cada `.sbatch`.

| Modelo | GPUs |
|---|---|
| Qwen2.5-32B-Instruct | 2 |
| Qwen2.5-72B-Instruct | 4 |
| Qwen3.6-27B | 2 (⚠️ ver nota abajo) |
| Qwen3-Next-80B-A3B-Instruct | 4 |

## Nota sobre imports entre scripts

Los scripts con prefijo numérico no se pueden importar con `import X` normal
(un nombre de módulo no puede empezar con dígito) -- se usa
`importlib.import_module("02_evaluar_licencias_llm")` en los 5 lugares donde
un script reutiliza lógica de otro (mismo truco que usan las migraciones de
Django/Alembic). El grafo real de dependencias:

```
04_asignar_licencia_llm.py  ──imports──▶  02_evaluar_licencias_llm.py (recortar_a_tokens)
                             ──imports──▶  03_juzgar_licencias_llm.py (DICC_LICENCIAS, DICCIONARIO_LICENCIAS_TXT)
03_juzgar_licencias_llm.py  ──imports──▶  02_evaluar_licencias_llm.py (recortar_a_tokens)
02_evaluar_licencias_llm.py ──imports──▶  compartido/evaluar_licencias_modo_server.py (PROMPT, parseo, combinar_detalle, etc.)
compartido/matizar_depende_llm.py ──imports──▶  compartido/evaluar_licencias_modo_server.py (MODELO, SALIDA, endpoint_localhost, texto_de_unidad)
```

`01_detectar_licencia.py` no depende de nada ni nadie depende de él en
tiempo de import (los demás scripts solo consumen su *salida*, el csv/jsonl).

## ⚠️ Cosas a tener en cuenta antes de usar esto

- **Qwen3.6-27B no sirve para este pipeline.** No es un release oficial de
  Qwen (es un checkpoint propio de otro proyecto del cluster, continued-pretrain
  sobre arquitectura Qwen3.5, multimodal) y no sigue instrucciones de chat de
  forma confiable -- repite el formato del prompt en vez de responder. Falla
  en el 98% de los casos en la tarea de asignar licencia. Queda documentado
  en `docs/RESULTADOS_comparacion_modelos.md` sección 0, no lo saqué del
  repo por si alguien quiere confirmarlo de nuevo, pero no usar sus
  resultados para nada real.
- **`llm.generate()` vs `llm.chat()`**: los 3 scripts batch (`02`, `03`, `04`)
  usan `llm.chat()`, no `llm.generate()` -- `generate()` manda el prompt como
  texto plano sin el chat template del modelo, lo que puede hacer que el
  modelo "continúe" el texto en vez de responder. Si agregás un script nuevo
  que llame a vLLM offline, usar `chat()`.
- **Un solo prompt largo puede tumbar un job entero** en modo batch (todos
  los prompts se validan antes de generar nada). Por eso `02`, `03` y `04`
  tienen un recorte de respaldo por tokens reales (`recortar_a_tokens`,
  definido en `02`) además del recorte por caracteres -- si agregás un prompt
  nuevo en batch, usarlo.
- **Ganador actual de la comparación**: Qwen3-Next-80B-A3B-Instruct (mejor
  juicio y mejor asignación de licencia, además el más rápido generando una
  vez cargado por ser MoE) -- detalle completo en
  `docs/RESULTADOS_comparacion_modelos.md`.

## Para profundizar

- `docs/PLAN_comparacion_modelos.md` -- qué se hizo y qué falta, fase por fase.
- `docs/RESULTADOS_comparacion_modelos.md` -- resultados completos de la
  comparación de 4 modelos × 2 temperaturas, en las 3 tareas.
- `docs/PROMPTS.md` -- los 4 prompts tal cual se usan.
