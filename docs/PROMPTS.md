# Prompts usados en el pipeline de licencias

Los 4 prompts que se les manda a los modelos, tal cual están en el código
(`compartido/evaluar_licencias_modo_server.py`, `03_juzgar_licencias_llm.py`,
`04_asignar_licencia_llm.py`). Todos se mandan vía `llm.chat([{"role": "user",
"content": prompt}], ...)` -- el chat template del modelo envuelve esto con
sus propias marcas de turno, no son parte del texto acá. `temperature=0` en
todos salvo donde se indique lo contrario (la comparación con `_temp0.7`).

## 1. Evaluar (SI / NO / DEPENDE)

Uno por cada unidad de texto (footer, subpágina, PDF) de una URL --
`compartido/evaluar_licencias_modo_server.py` / `02_evaluar_licencias_llm.py`.

```
Eres un asistente legal. Te paso un texto encontrado en el sitio de origen de un contenido, relacionado con su uso permitido -- puede ser una licencia (ej. Creative Commons, Apache), términos y condiciones de uso, una política de privacidad/datos, u otro tipo de texto legal del sitio.
Responde SOLO en este formato:
VEREDICTO: SI | NO | DEPENDE
JUSTIFICACION: <una o dos frases explicando por qué, citando la cláusula relevante>

Texto encontrado:
{licencia}
```

`{licencia}` = texto de esa unidad, recortado a 5000 caracteres (y además
por tokens reales antes de mandarlo, ver `recortar_a_tokens`).

## 2. Matizar (nivel 0-5, solo si el veredicto combinado dio DEPENDE)

```
Eres un asistente legal. Ya evaluaste este texto legal y contestaste DEPENDE sobre si se puede usar el contenido. Ahora necesito que le pongas un nivel del 0 al 5 de qué tan cerca está de poder usarse:
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
```

`{licencia}` = texto de las unidades que individualmente dieron DEPENDE
(no todas las de la URL). `{justificacion_llm}` = la justificación
combinada del paso 1.

## 3. Juzgar (BIEN / MAL / PARCIAL)

Compara el veredicto de un evaluador contra la clasificación manual de
`referencia/licencias_original_a_mano.xlsx` -- `03_juzgar_licencias_llm.py`, siempre con
Llama-3.3-70B-Instruct.

```
Eres un revisor legal. Te paso dos evaluaciones sobre si se puede usar el contenido de una fuente, según su licencia: una clasificación hecha a mano por un humano (referencia), y un veredicto que dio otro modelo de lenguaje (a evaluar).

Además tenés un diccionario de licencias conocidas, con su descripción en términos de qué permiten y qué restringen -- usalo como referencia cuando el comentario manual o la justificación mencionen alguna de estas licencias, para confirmar si el veredicto del modelo (SI/NO/DEPENDE) es consistente con lo que esa licencia realmente permite.

Diccionario de licencias:
{diccionario_licencias}

Tu tarea es juzgar si el veredicto del modelo coincide con la clasificación manual (y, cuando aplique, con lo que indica el diccionario de licencias de arriba).
Responde SOLO en este formato:
VEREDICTO: BIEN | MAL | PARCIAL
EXPLICACION: <una o dos frases explicando por qué>

Comentarios de licencia (referencia, escritos a mano):
{comentarios_manual}

Veredicto del modelo a evaluar:
Veredicto: {veredicto_llm}
Justificación: {justificacion_llm}
```

`{diccionario_licencias}` = las 20 licencias de `referencia/dicc_lic.json` (nombre +
descripción, ~2575 caracteres fijos). `{comentarios_manual}` = columna
"Comentarios Licencia" del xlsx de referencia. `{veredicto_llm}` /
`{justificacion_llm}` = lo que respondió el evaluador en el prompt 1.

## 4. Asignar licencia del diccionario

Un prompt por URL (no por unidad) -- lee el texto completo combinado de
`comentarios_licencia` en `licencias_detectadas.csv`. `04_asignar_licencia_llm.py`.

```
Eres un asistente legal especializado en licencias de datos. Te paso el texto legal encontrado en el sitio de origen de un contenido -- puede ser una licencia, términos y condiciones de uso, política de privacidad/datos, u otro texto legal del sitio.

Tu tarea es asignar la licencia que MEJOR corresponda de esta lista de licencias conocidas, según lo que el texto REALMENTE permite y restringe (no solo por el nombre que aparezca literal, si es que aparece):

{diccionario_licencias}

Si el texto no corresponde claramente a ninguna de estas licencias (ej. es solo una política de privacidad sin mención de licencia de datos, o el texto es insuficiente o ambiguo), respondé "Ninguna coincide".

Responde SOLO en este formato:
LICENCIA: <nombre EXACTO de la lista de arriba, o "Ninguna coincide">
JUSTIFICACION: <una o dos frases citando la cláusula relevante>

Texto encontrado:
{texto}
```

`{diccionario_licencias}` = mismo bloque que el prompt 3. `{texto}` = el
texto completo de la URL (recortado a 5000 caracteres + recorte por tokens
reales), no por unidad.
