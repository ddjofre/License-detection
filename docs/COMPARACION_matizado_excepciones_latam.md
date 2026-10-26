# Comparación del matizado: con vs. sin contexto de `excepciones_latam_v1.csv`

Objetivo: medir si agregar el bloque de contexto legal por país (ver
`PROMPT_MATIZ` / `bloque_contexto_pais` en
`compartido/evaluar_licencias_modo_server.py`) cambia de verdad el nivel
0-5 que el modelo le pone a un DEPENDE, y en qué dirección.

## Qué se comparó

Mismo modelo (Qwen3-Next-80B-A3B-Instruct), misma temperatura (0), dos
corridas distintas:

| | Sin contexto de país | Con contexto de país |
|---|---|---|
| Archivo | `salida/qwen3-next-80b-a3b-instruct/licencias_evaluadas_qwen3-next-80b-a3b-instruct.csv` | `salida/licencias_evaluadas_qwen3-next-80b-a3b.csv` |
| Corrida | Comparación de 4 modelos (cmp4), 2026-10-09 | Producción, 2026-10-19 |
| URLs | 341 (subconjunto fijo) | 467 (jsonl completo) |
| `PROMPT_MATIZ` | Versión de antes de hoy (sin `{contexto_pais}`) | Versión con `{contexto_pais}` |

Se cruzaron las URLs presentes en **ambas** corridas (341, la intersección
la define la corrida chica) y se filtraron las que dieron **DEPENDE en las
dos** (83). De esas, se separó:

- **Grupo con match de país** (16 URLs): el host matchea un dominio de
  `referencia/excepciones_latam_v1.csv` (`contexto_pais_de_url()` devuelve
  algo) -- a estas SÍ les cambió el prompt de matiz.
- **Grupo control, sin match** (67 URLs): el host no matchea ningún país
  de la lista -- a estas el prompt de matiz quedó **byte a byte idéntico**
  entre las dos corridas (`{contexto_pais}` se resuelve vacío en ambas).

## Primero, una advertencia necesaria: hay ruido de fondo

Antes de mirar el efecto del contexto, hay que descartar que el modelo
simplemente no sea 100% determinista entre corridas aunque `temperature=0`
(nondeterminismo de kernels/batching en GPU, ya documentado para vLLM). Dos
señales de esto, independientes del cambio de hoy:

- **9 URLs cambiaron de `veredicto_llm`** (SI/NO/DEPENDE/ERROR) entre las
  dos corridas, en el paso de **evaluar** -- que usa el mismo `PROMPT` sin
  ningún cambio hoy. Una es real (`eppetroecuador.ec`: NO -> ERROR, por el
  límite de tokens del servidor, no relacionado); las otras 8 son
  oscilaciones DEPENDE<->SI entre corridas con exactamente el mismo prompt.
- **En el grupo control** (67 URLs DEPENDE en ambas, prompt de matiz
  idéntico), **12 de 67 (18%) cambiaron de nivel** igual, sin que nada en
  el prompt haya cambiado. Direcciones mixtas: 9 subieron, 3 bajaron.

Esto pone un piso de ruido: cualquier diferencia por debajo de ~15-20% de
cambios, sin dirección consistente, no se puede atribuir al contexto de
país con confianza.

## El efecto del contexto de país

| Grupo | URLs | Cambiaron de nivel | Dirección |
|---|---|---|---|
| Control (sin match de país) | 67 | 12 (18%) | mixta (9 arriba, 3 abajo) |
| Con contexto de país | 16 | 13 (**81%**) | **13/13 hacia arriba** (ninguna bajó) |

La tasa de cambio en el grupo con contexto (81%) es muy superior al ruido
de fondo (18%), y a diferencia del control, **el 100% de los cambios van
en la misma dirección**: el modelo sube el nivel (más inclinado a que SÍ
se puede usar) cuando el país tiene una norma o una licencia por defecto
favorable (CC0, Open Data, exclusión de textos oficiales, reproducción
permitida). Eso es justo el efecto buscado al agregar el bloque -- y es
consistente, no ruido.

## Detalle de los 16 casos con contexto de país

| URL | País | Nivel sin ctx | Nivel con ctx | Cambio |
|---|---|---|---|---|
| `www.pj.gob.pe` | Perú | 2 | 3 | ↑ |
| `www.indap.gob.cl/` | Chile | 4 | 4 | = |
| `www.mcj.go.cr` | Costa Rica | 1 | 2 | ↑ |
| `mng.mincultura.gov.co/` | Colombia | 2 | 3 | ↑ |
| `www.gob.pe/busquedas` | Perú | 2 | 3 | ↑ |
| `www.gob.pe/busquedas?term=repositorio...` | Perú | 2 | 3 | ↑ |
| `www.gob.pe/institucion/servir/noticias` | Perú | 2 | 3 | ↑ |
| `www.mincultura.gov.co/` | Colombia | 2 | 2 | = |
| `www.sinabi.go.cr/biblioteca.../index.aspx` | Costa Rica | 2 | 3 | ↑ |
| `www.archivonacional.gob.cl/` | Chile | 2 | 3 | ↑ |
| `www.memoriachilena.gob.cl/602/...` | Chile | 2 | 2 | = |
| `www.archivogeneral.gov.co/` | Colombia | 2 | **4** | ↑↑ |
| `www.cultura.gob.cl/noticias/` | Chile | 2 | 3 | ↑ |
| `www.ramajudicial.gov.co/web/guest/...` | Colombia | 0 | 1 | ↑ |
| `www.pj.gov.py` | Paraguay | 2 | 3 | ↑ |
| `www.gob.pe/inei/` | Perú | 2 | 3 | ↑ |

Las 3 que no cambiaron (`indap.gob.cl`, `mincultura.gov.co`,
`memoriachilena.gob.cl`) ya estaban en nivel alto/medio y el modelo las
dejó igual, mencionando el contexto pero sin que alterara la conclusión
(ver `indap.gob.cl` abajo -- ya decía "probablemente usable" sin el
contexto).

## Ejemplos de cómo cambió el razonamiento (no solo el número)

**`archivonacional.gob.cl` (Chile, CC0 por defecto):**
- *Sin contexto:* "El texto no otorga permiso explícito [...] solo
  menciona que rige la ley chilena de propiedad intelectual, lo que deja
  ambigüedad [...]. Requiere verificación específica."
- *Con contexto:* "[...] al ser un organismo estatal chileno y
  considerando el contexto legal de que el Estado puede liberar obras bajo
  licencias abiertas (como CC0), es probable que algunos contenidos sean
  reutilizables [...]"

**`archivogeneral.gov.co` (Colombia, textos oficiales reproducibles) --
el salto más grande (2 → 4):**
- *Sin contexto:* "[...] contiene elementos de formato dañado,
  inconsistencias y una cláusula de 'copia no controlada' que genera duda
  [...]. No hay autorización explícita."
- *Con contexto:* "[...] según la legislación colombiana, los textos
  normativos de entidades públicas suelen ser reproducibles bajo el
  principio de acceso a la información pública [...]. Aunque no hay una
  autorización explícita de uso, su naturaleza pública y el marco legal
  colombiano lo hacen probablemente usable."

**`pj.gov.py` (Paraguay, textos judiciales excluidos de derecho de
autor):**
- *Sin contexto:* "El texto solo contiene una declaración genérica de
  derechos reservados [...] la falta de autorización explícita lo hace más
  inclinado a no poder usarse."
- *Con contexto:* "[...] al ser un contenido de la Corte Suprema de
  Justicia de Paraguay, el contexto legal del país indica que los textos
  judiciales oficiales están excluidos de protección de derechos de autor,
  lo que inclina hacia un uso permitido con atribución [...]"

El modelo no ignora el texto encontrado en el sitio (sigue sin asumir una
licencia que el texto no menciona), pero usa el contexto como un segundo
factor que empuja el nivel hacia arriba cuando el texto del sitio es
ambiguo -- justo el comportamiento que pedía la instrucción agregada al
prompt ("considerarse como referencia adicional [...] sin tomarlo como
definitivo").

## Limitaciones de esta comparación

- No es un experimento controlado puro: las dos corridas no son la MISMA
  corrida con un solo cambio -- son corridas distintas (341 vs. 467 URLs,
  10 días de diferencia), así que el ruido de fondo del 18% en el control
  es la mejor estimación disponible, no una cota exacta.
- El grupo con contexto es chico (16 URLs) -- suficiente para ver una señal
  clara (81% vs. 18%, 100% misma dirección), pero no para estadística fina
  por país.
- Solo cubre los 6 países que aparecieron en este subconjunto de 341/467
  URLs (Perú, Chile, Colombia, Costa Rica, Paraguay) de los 20 que tiene
  `excepciones_latam_v1.csv` -- el resto (Argentina, Brasil, México,
  Uruguay, etc.) no tuvo ningún DEPENDE en este conjunto de URLs para
  evaluar.

## Conclusión

El contexto de `excepciones_latam_v1.csv` sí está teniendo el efecto
buscado: para las URLs de dominios gubernamentales de LatAm que dieron
DEPENDE, el modelo sube el nivel de uso de forma consistente (81% de los
casos, siempre hacia arriba) cuando el país tiene una norma o licencia por
defecto favorable -- muy por encima del ruido de fondo del 18% que existe
igual sin ningún cambio de prompt. El razonamiento que cita el modelo en la
`explicacion_nivel_llm` confirma que está usando el bloque nuevo como
justificación, no solo cambiando el número sin explicación.
