# Resultados: comparación de modelos Qwen para licencias

Comparación de 4 modelos en 3 tareas (evaluar SI/NO/DEPENDE, juzgar contra
referencia manual, asignar licencia de un diccionario cerrado), sobre el
mismo subconjunto fijo de **341 URLs** (evaluar/juzgar) o las **467 URLs
completas** (asignar licencia, incluye las 126 "Alianza"). Metodología
completa en `PLAN_comparacion_modelos.md`.

**Esta es la versión final, post-fix** (2026-10-05/06) -- reemplaza una
corrida anterior con un bug real, ver sección 0.

## 0. El bug que invalidó la primera corrida (y cómo se arregló)

Los 3 scripts batch (`02_evaluar_licencias_llm.py`, `03_juzgar_licencias_llm.py`,
`04_asignar_licencia_llm.py`) llamaban a `llm.generate(prompts, ...)` -- un
endpoint de **completado de texto plano**, sin las marcas de turno del chat
template del modelo. Sin esas marcas, un modelo puede terminar *continuando*
el texto en vez de responder como asistente. Se confirmó con Qwen3-Next-80B-A3B
en la tarea de asignar licencia: el **56%** de las respuestas no seguían el
formato pedido -- el modelo citaba fragmentos del texto legal como si fuera
parte del mismo documento, en vez de contestar.

**Fix**: cambiar `llm.generate(prompts, ...)` por
`llm.chat([{"role": "user", "content": p}], ...)`, que sí aplica el chat
template. Confirmado con una prueba directa (`inspect.signature`) de que
`generate()` no lo aplica y `chat()` sí. El evaluador de producción
(`compartido/evaluar_licencias_modo_server.py`) nunca tuvo este problema porque le pega a un
servidor vLLM por `/v1/chat/completions`, que aplica el template del lado
del servidor.

**Efecto colateral del fix**: reveló que **Qwen3.6-27B no es un modelo
instruct confiable para estas tareas**. Antes del fix (con texto plano) daba
resultados que *parecían* razonables por pura coincidencia; con `chat()`
aplicado correctamente, el modelo repite literalmente el formato del prompt
(`VEREDICTO: SI | NO | DEPENDE`, placeholders sin completar) en vez de
responder, y en la tarea de asignar licencia falla en el **98% de los
casos**. Es un checkpoint propio de LatamGPT (continued-pretrain sobre
arquitectura Qwen3.5, multimodal), no un release oficial de Qwen con SFT de
chat -- queda **descartado de las 3 tareas**, no es un problema de prompt.

Todos los números de acá en adelante son **post-fix**, con Qwen3.6-27B
incluido solo como referencia de que sigue roto.

## 1. Tarea 1: Evaluar SI/NO/DEPENDE (341 URLs, 204 unidades de texto)

| Modelo | GPUs | Carga | Evaluar | Matizar | **Total** |
|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 2 | 94.1s | 18.1s | 13.4s | **2:07** |
| Qwen2.5-72B-Instruct | 4 | 145.5s | 24.5s | 14.7s | **3:06** |
| Qwen3.6-27B ⚠️ | 2 | 237.0s | 36.2s | 0.0s | 4:35 |
| Qwen3-Next-80B-A3B-Instruct | 4 | 470.0s | 5.6s | 3.3s | **8:01** |

| Modelo | SI | NO | DEPENDE | SIN_TEXTO |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 24 (7.0%) | 29 (8.5%) | 130 (38.1%) | 158 (46.3%) |
| Qwen2.5-72B-Instruct | 44 (12.9%) | 22 (6.5%) | 117 (34.3%) | 158 (46.3%) |
| Qwen3.6-27B ⚠️ | 183 (53.7%) | **0** | **0** | 158 (46.3%) |
| Qwen3-Next-80B-A3B-Instruct | 78 (22.9%) | 15 (4.4%) | 90 (26.4%) | 158 (46.3%) |

Qwen3.6-27B: 100% SI en las 183 URLs con texto, 0% NO/DEPENDE -- el "SI" es
un artefacto del parseo (el regex agarra el "SI" de la propia línea de
instrucciones que el modelo repite: `VEREDICTO: SI | NO | DEPENDE`), no un
juicio real. Confirmado leyendo las justificaciones crudas: empiezan con el
placeholder literal del prompt (`<una o dos frases...>`) seguido de
razonamiento en inglés sin relación con el formato pedido.

Qwen3-Next-80B-A3B vuelve a confirmar la ventaja de velocidad del MoE una
vez cargado (5.6s para evaluar 204 unidades, contra 18-36s de los demás) --
aunque acá cargó más lento que en la corrida anterior (470s vs 503s, dentro
del ruido normal del cluster compartido).

### 1b. Matizado (nivel 0-5, solo filas DEPENDE)

0 = definitivamente NO se puede usar, 5 = definitivamente SI. Sobre las
filas que cada modelo clasificó como DEPENDE en la tabla anterior (el total
de filas matizadas varía por modelo, porque varía cuántas dieron DEPENDE).

| Modelo | DEPENDE (n) | 0 | 1 | 2 | 3 | 4 | 5 | **Promedio** |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 130 | 2 (1.5%) | 22 (16.9%) | 32 (24.6%) | 23 (17.7%) | 51 (39.2%) | 0 | **2.76** |
| Qwen2.5-72B-Instruct | 117 | 2 (1.7%) | 0 | 43 (36.8%) | 72 (61.5%) | 0 | 0 | **2.58** |
| Qwen3.6-27B ⚠️ | 0 | — | — | — | — | — | — | N/A (nunca da DEPENDE, ver arriba) |
| Qwen3-Next-80B-A3B-Instruct | 90 | 10 (11.1%) | 30 (33.3%) | 26 (28.9%) | 17 (18.9%) | 7 (7.8%) | 0 | **1.79** |

**Ninguno de los tres usó nunca el nivel 5** (definitivamente SI) para matizar
un DEPENDE -- coherente, un DEPENDE que terminara en "definitivamente SI"
debería haber sido directamente SI en el primer veredicto.

**Qwen3-Next-80B-A3B es notablemente más conservador**: promedio 1.79, con
el 73.3% de sus DEPENDE concentrados en 0-2 (más cerca de NO). Qwen2.5-32B es
el más permisivo (promedio 2.76, 39.2% en nivel 4). Qwen2.5-72B queda en el
medio pero con una distribución rara: todo concentrado en 2 y 3, nunca usa
1 ni 4 -- parece menos granular/más binario al matizar que los otros dos,
pese a ser el que más veces elige DEPENDE en el veredicto inicial.

## 2. Tarea 2: Juicio de Llama-3.3-70B (166 filas juzgables de 341)

166 = 341 evaluadas − 158 sin texto − 14 sin fila de referencia manual − 3
excluidas por mencionar "alianza". El juez tiene acceso al diccionario de
licencias (`referencia/dicc_lic.json`) además del comentario manual.

| Modelo | BIEN | MAL | **% BIEN** |
|---|---|---|---|
| Qwen2.5-32B-Instruct | 37 | 129 | 22.3% |
| Qwen2.5-72B-Instruct | 39 | 127 | 23.5% |
| Qwen3.6-27B ⚠️ | 31 | 135 | 18.7% |
| **Qwen3-Next-80B-A3B-Instruct** | **49** | 117 | **29.5%** |

Ningún PARCIAL en ninguno de los 4 (antes del fix había 2-29 por modelo) --
el juez también quedó más tajante al recibir prompts bien formados.

**El ranking se invirtió por completo respecto a la corrida con bug**:
antes Qwen3.6-27B parecía ganar (23%) -- era justo el modelo roto, cuyos
veredictos tajantes casualmente "sonaban bien". Con el fix, **Qwen3-Next-80B-A3B
gana claro** (29.5%, ~6-11 puntos por encima de los otros dos buenos) y
Qwen3.6-27B queda último, coherente con que sus veredictos son ruido.

## 3. Tarea 3: Asignar licencia del diccionario (467 URLs, 261 con texto)

| Modelo | GPUs | Carga | Asignar | **Total** |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 2 | 118.0s | 29.7s | **2:29** |
| Qwen2.5-72B-Instruct | 4 | 141.5s | 35.5s | **2:58** |
| Qwen3.6-27B ⚠️ | 2 | 230.2s | 54.2s | 4:46 |
| Qwen3-Next-80B-A3B-Instruct | 4 | 470.8s | 9.7s | **8:02** |

| Modelo | Licencia válida | "Ninguna coincide" | "(sin match)" |
|---|---|---|---|
| Qwen2.5-32B-Instruct | 62 (23.8%) | 199 (76.2%) | 0 |
| Qwen2.5-72B-Instruct | 75 (28.7%) | 184 (70.5%) | 2 (0.8%) |
| Qwen3.6-27B ⚠️ | **0** | 4 (1.5%) | **257 (98.5%)** |
| **Qwen3-Next-80B-A3B-Instruct** | **121 (46.4%)** | 138 (52.9%) | 2 (0.8%) |

**Qwen3-Next-80B-A3B gana por lejos**: asigna una licencia válida en el
46.4% de las URLs con texto, contra 24-29% de los otros dos buenos.
Qwen3.6-27B confirma que está completamente roto (98.5% de fallos de
formato, peor que en evaluar porque este prompt es más largo/complejo).

## 4. Resumen final

| Eje | Ganador |
|---|---|
| % BIEN del juez | **Qwen3-Next-80B-A3B** (29.5%) |
| % licencia válida asignada | **Qwen3-Next-80B-A3B** (46.4%) |
| Velocidad una vez cargado (evaluar + asignar) | **Qwen3-Next-80B-A3B** (MoE, ~3B params activos) |
| Tiempo total con carga incluida (corrida puntual) | Qwen2.5-32B (el más rápido de punta a punta) |
| Costo en GPU-segundos (corrida puntual) | Qwen2.5-32B (el más barato) |
| Confiabilidad de formato | Qwen2.5-32B y Qwen3-Next-80B-A3B empatan (Qwen2.5-72B muy cerca) |

**Conclusión**: con el bug corregido, **Qwen3-Next-80B-A3B-Instruct es el
mejor candidato** en las dos métricas de calidad (juicio y asignación de
licencia) y además el más rápido generando una vez cargado -- el costo es
un tiempo de carga más largo y 4 GPUs en vez de 2, que en una corrida batch
puntual como esta pesa, pero en un server persistente de producción (carga
una sola vez) dejaría de ser una desventaja real.

**Qwen3.6-27B queda descartado** de las 3 tareas -- no es un problema de
prompt, el checkpoint no sigue instrucciones de chat de forma confiable.

---

# Segunda versión del experimento: temperatura 0.7

Mismos 4 modelos, mismo subconjunto/corpus, mismos prompts y mismo juez --
única diferencia: `CMP_TEMPERATURE=0.7` / `ASIG_TEMPERATURE=0.7` en vez de 0
(determinístico) para los pasos de evaluar/matizar y asignar licencia. El
juez (`03_juzgar_licencias_llm.py`) sigue siempre a `temperature=0` --
hardcodeado, no varía entre corridas, para que el criterio de juicio en sí
no sea una variable.

## 5. Resultados a temperatura 0.7

### 5.1. Evaluar SI/NO/DEPENDE (341 URLs, 204 unidades)

| Modelo | GPUs | Carga | Evaluar | Matizar | **Total** |
|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 2 | 123.4s | 18.4s | 12.8s | **2:36** |
| Qwen2.5-72B-Instruct | 4 | 357.0s | 25.7s | 16.2s | **6:41** |
| Qwen3.6-27B ⚠️ | 2 | 237.2s | 36.7s | 0.0s | 4:36 |
| Qwen3-Next-80B-A3B-Instruct | 4 | 572.3s | 6.0s | 3.3s | **9:44** |

| Modelo | SI | NO | DEPENDE | SIN_TEXTO |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 38 (11.1%) | 28 (8.2%) | 117 (34.3%) | 158 (46.3%) |
| Qwen2.5-72B-Instruct | 45 (13.2%) | 22 (6.5%) | 116 (34.0%) | 158 (46.3%) |
| Qwen3.6-27B ⚠️ | 183 (53.7%) | **0** | **0** | 158 (46.3%) |
| Qwen3-Next-80B-A3B-Instruct | 78 (22.9%) | 16 (4.7%) | 89 (26.1%) | 158 (46.3%) |

### 5.2. Matizado (nivel 0-5, solo filas DEPENDE)

| Modelo | DEPENDE (n) | 0 | 1 | 2 | 3 | 4 | 5 | **Promedio** |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 117 | 3 (2.6%) | 16 (13.7%) | 34 (29.1%) | 32 (27.4%) | 32 (27.4%) | 0 | **2.63** |
| Qwen2.5-72B-Instruct | 116 | 2 (1.7%) | 0 | 44 (37.9%) | 60 (51.7%) | 10 (8.6%) | 0 | **2.66** |
| Qwen3.6-27B ⚠️ | 0 | — | — | — | — | — | — | N/A |
| Qwen3-Next-80B-A3B-Instruct | 89 | 8 (9.0%) | 31 (34.8%) | 28 (31.5%) | 12 (13.5%) | 10 (11.2%) | 0 | **1.83** |

### 5.3. Juicio de Llama-3.3-70B (166 filas)

| Modelo | BIEN | MAL | PARCIAL | **% BIEN** |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 42 | 124 | 0 | 25.3% |
| Qwen2.5-72B-Instruct | 39 | 127 | 0 | 23.5% |
| Qwen3.6-27B ⚠️ | 31 | 133 | 2 | 18.7% |
| **Qwen3-Next-80B-A3B-Instruct** | **50** | 116 | 0 | **30.1%** |

### 5.4. Asignar licencia del diccionario (467 URLs, 261 con texto)

| Modelo | GPUs | Carga | Asignar | **Total** |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 2 | 127.6s | 30.6s | **2:39** |
| Qwen2.5-72B-Instruct | 4 | 99.0s | 35.3s | **2:15** |
| Qwen3.6-27B ⚠️ | 2 | 238.9s | 54.8s | 4:55 |
| Qwen3-Next-80B-A3B-Instruct | 4 | 562.0s | 10.2s | **9:34** |

| Modelo | Licencia válida | "Ninguna coincide" | "(sin match)" |
|---|---|---|---|
| Qwen2.5-32B-Instruct | 62 (23.8%) | 199 (76.2%) | 0 |
| Qwen2.5-72B-Instruct | 74 (28.4%) | 182 (69.7%) | 5 (1.9%) |
| Qwen3.6-27B ⚠️ | 5 (1.9%) | 8 (3.1%) | 248 (95.0%) |
| **Qwen3-Next-80B-A3B-Instruct** | **122 (46.7%)** | 137 (52.5%) | 2 (0.8%) |

**El ranking se mantiene idéntico al de temperatura 0**: Qwen3-Next-80B-A3B
gana en juicio (30.1%) y en asignación de licencia (46.7%), Qwen3.6-27B
sigue roto (95% de fallos de formato). Los números individuales de cada
modelo se mueven un poco (esperable, es una corrida no determinística), pero
nada cambia de orden.

## 6. Comparación de temperaturas (estabilidad: temp 0 vs. temp 0.7)

Qué tan seguido un modelo da el MISMO resultado para la MISMA URL entre las
dos corridas (temp 0 vs temp 0.7) -- mide estabilidad/consistencia del
modelo, no calidad. Un modelo muy estable no es necesariamente mejor, pero
si va a producción con muestreo (`temperature > 0`, por ejemplo para generar
variedad), importa saber cuánto puede cambiar el resultado de una corrida a
otra.

| Modelo | Evaluar (341) | Juez (166) | Asignar licencia (467) |
|---|---|---|---|
| Qwen2.5-32B-Instruct | 324/341 (95.0%) | 151/166 (91.0%) | 467/467 (**100.0%**) |
| Qwen2.5-72B-Instruct | 332/341 (97.4%) | 150/166 (90.4%) | 462/467 (98.9%) |
| Qwen3.6-27B ⚠️ | 341/341 (**100.0%**) | 153/166 (92.2%) | 325/467 (69.6%) |
| Qwen3-Next-80B-A3B-Instruct | 340/341 (99.7%) | 155/166 (93.4%) | 446/467 (95.5%) |

**Lecturas:**

- **Evaluar y asignar licencia son bastante estables** para los 3 modelos
  buenos (95-100%) -- la temperatura 0.7 casi no cambia el veredicto final,
  probablemente porque el combinado "el peor veredicto de las unidades gana"
  (`combinar_detalle`) amortigua el ruido de muestreo de unidades
  individuales.
- **El juez es el menos estable de los tres** (90-93% en los 4 modelos) --
  tiene sentido, es el único de los tres pasos que siempre corrió a
  `temperature=0`, así que el 7-10% de cambio viene enteramente de que el
  INPUT que juzga (el veredicto del evaluador a 0.7) varió, no de que el
  juez mismo sea inconsistente.
- **Qwen3.6-27B es 100% estable evaluando pero el menos estable asignando
  licencia** (69.6%, el peor de los 4) -- coherente con que su output ahí es
  básicamente texto aleatorio sin relación con la pregunta: al no estar
  realmente "razonando", lo único que cambia entre corridas es qué
  fragmento de texto decide repetir, que varía mucho más con temperatura que
  una decisión real.
- Qwen3-Next-80B-A3B (el ganador en calidad) también es sólido en
  estabilidad (95.5-99.7%), menos en asignar licencia que Qwen2.5-32B/72B
  pero por encima de Qwen3.6-27B.

## 7. Pendiente

- [ ] Segundo juez (Gemma-4-31B) para contrastar el ranking con un juez de
  otra familia -- nunca se armó, sigue en el PLAN.
- [ ] Revisión manual de una muestra de filas MAL / "Ninguna coincide" de
  Qwen3-Next-80B-A3B, para entender si los casos que no asigna licencia son
  genuinamente ambiguos o margen de mejora real.
- [ ] Limpiar las filas viejas (pre-fix) de `tiempos.csv` y
  `tiempos_asignacion.csv` -- quedaron mezcladas con las nuevas.
- [ ] Si se decide usar un modelo con `temperature > 0` en producción,
  correr más de 2 muestras por URL para medir estabilidad con más
  confianza -- 2 puntos (0 y 0.7) alcanzan para una primera lectura, no para
  una conclusión robusta.
