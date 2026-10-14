# Plan: comparar evaluadores LLM para licencias + asignar licencia por diccionario

Documento de seguimiento para ir marcando qué está hecho y qué falta. No reemplaza
los docstrings de cada script (ahí está el detalle técnico) -- esto es el mapa de
las fases y el orden.

## Fase 0 — Detección y preparación del corpus (LISTO)

- [x] `01_detectar_licencia.py` corrido sobre el corpus completo -> `salida/licencias_detectadas.csv` / `.jsonl` (467 URLs)
- [x] Distinguir "Error de descarga" (falló la bajada) de "No encontrada" (se revisó y no había licencia) en `01_detectar_licencia.py` -- antes quedaban mezclados bajo el mismo label
- [x] `_urls_ya_procesadas()` excluye las filas "Error de descarga" del dedup, para que se reintenten solas en la próxima corrida
- [x] Filtradas las URLs "Alianza" (`Licencia == "Alianza"` en `referencia/licencias_original_a_mano.xlsx`, 126 URLs, algunas con varias URLs pegadas en la misma celda) desde `licencias_detectadas.jsonl` -> **`salida/comparacion_modelos/subset_urls.jsonl`** (341 URLs, el subconjunto fijo para todo lo que sigue)
- [x] Verificado que el subset no quedó sesgado: distribución SI/NO/DEPENDE/NO_ENCONTRADO casi igual entre el total y el subset (ver `salida/comparacion_modelos/subset_urls.csv` para inspección a ojo)

## Fase 1 — Comparar evaluadores Qwen (calidad + tiempo)

- [x] Fusionado evaluar + matizar en una sola pasada (`compartido/evaluar_licencias_modo_server.py`) -- ya no hace falta correr `compartido/matizar_depende_llm.py` aparte para corridas nuevas (queda solo para backfill de los csv viejos de producción)
- [x] Extraída `combinar_detalle()` de `evaluar_unidades()` para no duplicar la lógica de combinar veredictos
- [x] Armado `02_evaluar_licencias_llm.py`: modo offline/batch (vLLM, sin servidor, carga el modelo una sola vez, dos tandas de `generate()` -- evaluar y matizar)
- [x] Soporte de `CMP_TEMPERATURE` (default 0) -- el nombre de archivo de salida agrega `_tempX` automáticamente si no es 0
- [x] Agregado `tiempos.csv` consolidado (una fila por corrida: modelo, temperatura, GPUs, tiempos de carga/evaluar/matizar, conteo de veredictos)
- [x] `.sbatch` armados para los 4 modelos, todos a `CMP_TEMPERATURE=0` (default) y apuntando a `subset_urls.jsonl`:
  - `slurm/comparar_evaluadores_qwen2.5-32b.sbatch` -- 2 GPUs
  - `slurm/comparar_evaluadores_qwen2.5-72b.sbatch` -- 4 GPUs
  - `slurm/comparar_evaluadores_qwen3.6-27b.sbatch` -- 2 GPUs
  - `slurm/comparar_evaluadores_qwen3-next-80b-a3b.sbatch` -- 4 GPUs

- [x] Lanzadas las 4 corridas base a `CMP_TEMPERATURE=0` (2026-10-01) -- en cola (`PD`) esperando GPU libre en `lascar` (nodo compartido, 8 A100 en total, ocupado por otros jobs del equipo en el momento del envío):
  - `qwen2.5-32b` -> job **13155** (2 GPUs)
  - `qwen2.5-72b` -> job **13156** (4 GPUs)
  - `qwen3.6-27b` -> job **13157** (2 GPUs)
  - `qwen3-next-80b-a3b` -> job **13158** (4 GPUs)

  Seguir avance: `squeue --me` (estado) / `tail -f slurm_logs/det_lic_cmp_*_<jobid>.out` (log de cada uno).

- [x] **Resultado de esa primera tanda: 2 de 4 terminaron OK, 2 fallaron.**
  - ✅ `qwen2.5-32b` (job 13155) y `qwen3.6-27b` (job 13157) -> `COMPLETED`.
  - ❌ `qwen2.5-72b` (job 13156) y `qwen3-next-80b-a3b` (job 13158) -> `FAILED` con
    `VLLMValidationError: ... prompt contains at least 4097 input tokens` (límite 4096).
    Causa: `MAX_CHARS_LICENCIA` recorta por CARACTERES (heurística ~3.5 char/token),
    no por tokens reales -- alcanzó para 32B/3.6-27B pero no para 72B/Next-80B con
    el mismo subconjunto. En modo batch, `generate()` valida TODOS los prompts antes
    de generar nada, así que un solo prompt pasado de largo tumba el job ENTERO
    (a diferencia del evaluador de producción, donde cada unidad es un request HTTP
    aparte y un error ahí no afecta a las demás).
- [x] **Fix aplicado** en `02_evaluar_licencias_llm.py`: recorte de respaldo por
  tokens reales (`recortar_a_tokens()`, usa `transformers.AutoTokenizer` del
  modelo que se va a cargar) antes de mandar cada prompt a `generate()` --
  además del recorte por caracteres de siempre, que se deja como primer filtro
  barato. Probado contra el tokenizer real de Qwen2.5-72B vía `srun` (recortó
  40.002 tokens a 4.064 sin tocar un texto corto de control).
- [x] **Relanzados los 2 que habían fallado** con el fix ya aplicado (2026-10-02):
  - `qwen2.5-72b` -> job 13441 -> cancelado y relanzado como **job 13449** (job-name `cmp2`)
  - `qwen3-next-80b-a3b` -> job 13442 -> cancelado y relanzado como **job 13450** (job-name `cmp4`)
  (job-names cortos a pedido del usuario: `cmp1`=32B, `cmp2`=72B, `cmp3`=Qwen3.6-27B,
  `cmp4`=Qwen3-Next-80B-A3B -- ver mapeo en cada `.sbatch`; los 2 primeros, jobs 13155
  y 13157, ya habían corrido con el nombre viejo `det_lic_cmp_*`, no hizo falta
  re-lanzarlos)

**Pendiente:**

- [ ] Confirmar que 13441 y 13442 terminan OK con el fix (revisar `.out`/`.err`)
- [ ] **Decidir qué temperaturas probar además de 0** -- sin definir todavía. Ojo: temperatura > 0 mide *estabilidad* de un modelo (cuánto varía su propio veredicto entre corridas), no es directamente comparable como "calidad" contra otro modelo -- si se agrega esta dimensión, pensar si es "cada modelo a 2-3 temperaturas" o solo para el/los modelo(s) finalista(s) de la Fase 2.
- [ ] Una vez decididas las temperaturas: duplicar/parametrizar los `.sbatch` existentes con `CMP_TEMPERATURE=<valor>` (mismo script, no hace falta tocar código)
- [ ] Lanzar las corridas extra de temperatura, si corresponde
- [ ] Revisar `salida/comparacion_modelos/tiempos.csv` -- tiempo de carga/evaluar/matizar y conteo de veredictos por modelo (y por temperatura, si aplica)

## Fase 2 — Juzgar con dos jueces (Llama-3.3-70B y Gemma-4-31B)

- [x] Patrón ya existente y probado para un juez (`03_juzgar_licencias_llm.py`, offline/batch, compara contra `referencia/licencias_original_a_mano.xlsx`)

**Pendiente:**

- [ ] **Parametrizar el modelo juez en `03_juzgar_licencias_llm.py`** -- hoy `MODELO` está hardcodeado a Llama-3.3-70B-Instruct (a diferencia de `EVAL_MODELO` en el evaluador, que sí es variable de entorno). Hace falta un `JUZGAR_MODELO` (o similar) antes de poder correr el mismo script con Gemma-4-31B-it sin duplicar código.
- [ ] Armar `.sbatch` de juicio con **Llama-3.3-70B** (2 GPUs) para cada `licencias_evaluadas_<modelo>.csv` que salga de la Fase 1
- [ ] Armar `.sbatch` de juicio con **Gemma-4-31B-it** (1 GPU estimada, a confirmar) para los mismos csv
- [ ] Decidir dónde vive cada `juicio_<juez>.csv` -- según lo charlado: adentro de la carpeta del modelo evaluador (`salida/<modelo>/juicio_llama.csv`, `salida/<modelo>/juicio_gemma4-31b.csv`), no en `comparacion_modelos/`
- [ ] Correr el juicio de los dos jueces sobre cada evaluador (y cada temperatura, si la Fase 1 terminó agregando esa dimensión)
- [ ] Consolidar % BIEN / MAL / MAS_O_MENOS por modelo (y por juez, y por temperatura si aplica) -- posiblemente un csv resumen aparte, tipo `comparacion_modelos/resumen_juicios.csv`
- [ ] Con tiempos (Fase 1) + resultados (Fase 2) ya cruzados: **elegir el modelo (y temperatura) ganador** para producción

## Fase 3 — Asignar licencia desde diccionario

Tarea distinta a las anteriores: no es "¿se puede usar?" (SI/NO/DEPENDE), sino
"¿cuál de las licencias conocidas de nuestro diccionario es esta?" -- un
matching/clasificación contra un vocabulario cerrado (nombre de licencia +
condiciones), a partir del texto que ya encontró el detector.

**Dónde ubicarla -- mi recomendación:** después de la Fase 2, no en el medio.
Motivo: esta fase también depende de un LLM, y conviene correrla con el modelo
ya elegido como ganador (mejor relación calidad/tiempo confirmada), en vez de
correrla 4 veces en paralelo con cada candidato de la comparación -- no es parte
de lo que se está benchmarkeando (no estamos comparando qué tan bien asignan
licencia los distintos Qwen, solo cuál evalúa mejor SI/NO/DEPENDE). Si más
adelante interesa *también* comparar qué tan bien asignan licencia los
distintos modelos, se puede repetir esta fase con el segundo o tercer mejor
candidato, pero como paso aparte, no bloqueando el resto.

**Pendiente (todo, falta armar):**

- [ ] Formalizar el diccionario de licencias (nombre + condiciones) en un formato que un script pueda leer (csv/json/yaml -- a definir)
- [ ] Armar el prompt: dado el texto encontrado por el detector (y quizás el veredicto SI/NO/DEPENDE ya calculado), que el modelo elija la licencia del diccionario que mejor matchea, o indique que no matchea ninguna
- [ ] Armar el script (reutilizando lo que ya existe de `compartido/evaluar_licencias_modo_server.py`/`02_evaluar_licencias_llm.py` para no duplicar: lectura de unidades, texto de unidad, etc.)
- [ ] Decidir si corre sobre las 341 URLs del subset o sobre las 467 completas (incluyendo alianzas) -- las alianzas also podrían necesitar una licencia asignada, a diferencia de la Fase 1-2 donde no aplicaban
- [ ] Correr con el modelo ganador de la Fase 2
- [ ] Agregar la columna `licencia_asignada` (y quizás `confianza`/`justificacion`) al resultado final consolidado

## Fase 2 (del pipeline general) — Correr en producción con la nueva lista de links

A diferencia de las Fases 0-3 (que son la comparación de modelos sobre el
subconjunto fijo de 341/467 URLs ya conocidas), esto es la corrida real:
procesar una lista NUEVA de links con el pipeline ya elegido.

**Decidido:**

- [x] **Alcance**: pipeline completo -- `01_detectar_licencia.py` (detectar)
  → `02_evaluar_licencias_llm.py` o `compartido/evaluar_licencias_modo_server.py`
  (evaluar SI/NO/DEPENDE, con el modelo ganador **Qwen3-Next-80B-A3B-Instruct**,
  ver `RESULTADOS_comparacion_modelos.md`) → `04_asignar_licencia_llm.py`
  (asignar licencia del diccionario), los tres con ese mismo modelo.

**Pendiente:**

- [ ] **De dónde sale la lista nueva de links -- A DEFINIR.** ¿Ya existe un
  archivo (`.csv`/`.xlsx`/`.md`) en algún lado, o hay que armarlo primero?
  Cuando esté, anotar acá la ruta.
- [ ] Una vez resuelta la lista: armar el/los `.sbatch` correspondientes
  (mismo patrón que `slurm/sbatch/`, pero apuntando a la lista nueva en vez
  de `subset_urls.jsonl`, y con el modelo fijo en Qwen3-Next-80B-A3B en vez
  de variar por modelo)
- [ ] Decidir dónde queda la salida de esta corrida -- ¿separada de
  `salida/` (que hoy es todo de la comparación de modelos), en una carpeta
  tipo `salida_produccion/`?

## Fase 3 (del pipeline general) — Conectar `excepciones_latam` al pipeline

`referencia/excepciones_latam.json`/`.csv` existe (una fila por país de LatAm:
dominios oficiales, si los textos legales/normativos están excluidos de
derecho de autor, ley de acceso a la información, portal de datos abiertos,
licencia por defecto de ese portal con su `licencia_dicc` ya mapeado a una
clave de `dicc_lic.json`) pero **todavía no lo lee ningún script** -- lo
armó el usuario aparte, en paralelo a esta comparación.

Por qué vale la pena conectarlo: en la comparación de modelos, la mayoría de
las URLs terminan en **"Ninguna coincide"** al asignar licencia (52-76%
según el modelo, ver `RESULTADOS_comparacion_modelos.md` sección 3) -- para
sitios gubernamentales de LatAm, `excepciones_latam` podría convertir una
parte de esos "Ninguna coincide" en una licencia real conocida (ej. por
tratarse de un texto oficial sin derecho de autor, o por venir del portal
nacional de datos abiertos con licencia declarada), sin depender de que el
LLM lo infiera solo del texto de la página.

**Pendiente -- diseño todavía sin definir:**

- [ ] **Cómo se cruza una URL con una fila de `excepciones_latam`** -- por
  dominio (`dominios` de cada país, ej. `gob.ar`), ¿extraído de la URL con
  qué lógica exactamente? Hay países con varios dominios oficiales en una
  misma fila (separados por espacio).
- [ ] **Dónde entra en la cascada de `04_asignar_licencia_llm.py`** -- ¿antes
  del LLM (si el dominio matchea y el país tiene `licencia_defecto` clara,
  asignar directo sin consultar al modelo) o como contexto adicional en el
  prompt junto al diccionario, dejando que el modelo decida con esa info de
  más?
- [ ] Respetar el `orden_revision` que ya define el propio archivo (`_meta`):
  primero portal de datos abiertos, después transparencia activa/texto
  oficial, por último "requiere solicitud" si no aplica nada -- no es un
  simple lookup plano, es una cascada con prioridad.
- [ ] Ojo con las filas `varia_por_conjunto = true` y los estados
  `por_confirmar` -- no tratarlos como una licencia definitiva sin más.
- [ ] Decidir si esto se usa en la comparación de modelos (Fases 0-3) o solo
  en la corrida de producción (Fase 2 de esta sección) -- probablemente
  esto último, mismo criterio que ya se usó para no repetir la Fase 3
  original (asignar licencia) con los 4 modelos candidatos.
