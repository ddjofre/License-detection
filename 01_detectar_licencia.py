"""
Módulo de DETECCIÓN de licencias: siempre se le da un link (o una lista de
links) y ubica dónde vive la licencia/términos del sitio o recurso. No opera
sobre HTML ya descargado por otro código -- el punto de entrada es siempre
una URL, nunca un `html` que otro scraper haya bajado antes.

Dos funciones públicas:

  1. `detectar_licencia(url, ...)` -- prueba en cascada varias estrategias,
     de más barata/confiable a más costosa/heurística:
       1. Adaptador de API de plataforma conocida (HuggingFace, GitHub,
          DSpace, CKAN).
       2. `.well-known/` (tdmrep.json, license.txt, /LICENSE) -- va antes
          de descargar la página porque no depende de su HTML.
       3. Descarga de la página.
       4. `rel="license"` / JSON-LD en la página ya descargada.
       5. Footer.
       6. Cualquier heading del body que matchee una palabra clave.
       7. Crawl de un nivel sobre candidatos fuera del footer.
       8. Página "Acerca de" / "Sobre nosotros" -- último recurso.
     Por default se detiene en la primera que encuentra algo; con
     `agotar_todas=True` prueba todas y junta los hallazgos de cada una.
     Con `directorio_salida` además baja el contenido completo de lo que
     encuentra: PDF convertido a Markdown, o página HTML convertida a
     Markdown (más los PDF que tenga adentro).

  2. `procesar_planilla(ruta_entrada, ...)` -- corre `detectar_licencia`
     sobre una LISTA de URLs leída de un archivo de entrada (.csv,
     .xlsx/.xls o .md), bajando el contenido completo por default, y arma
     un .csv de salida con columnas
     `url,licencia,comentarios_licencia,via,url_referencia` (o .xlsx, si se
     pasa `ruta_salida` con esa extensión), más un `.jsonl` aparte
     (`<ruta_salida>.jsonl` por default) que relaciona cada url con el
     detalle de los archivos descargados (ver
     `metadatos_archivos_de_resultado`) y con la lista de unidades de
     texto por separado, sin concatenar (ver `unidades_de_resultado`) --
     pensada para evaluar cada documento legal por su cuenta en vez de un
     solo bloque de texto. Reanudable: si el archivo de
     salida ya existe, las URLs que ya tengan fila ahí se saltean. Uso
     desde la terminal:

         python 01_detectar_licencia.py entrada.xlsx --salida licencias.csv

Uso como módulo desde cualquier scraper:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "detectar_licencias"))
    from detectar_licencia import detectar_licencia, procesar_planilla

    resultado = detectar_licencia("https://huggingface.co/datasets/algo", contexto="mi-scraper")
    print(resultado["via"], resultado["hallazgos"])

Las palabras clave de detección (PALABRAS_CLAVE_DEFAULT) son ES/EN y se
comparan sin tildes ni mayúsculas; se pueden extender o pisar por completo
pasando `palabras_clave` a cualquiera de las dos funciones de arriba.

Este módulo NO decide dedup más allá de lo que ya hace `procesar_planilla`
(dedup por `url`, igual que el resto del repo) -- si se usa
`detectar_licencia` suelta desde otro scraper, la persistencia y el dedup
quedan del lado de ese scraper.
"""

import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import markdownify
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scraping" / "extras"))
from pdf_a_texto import convertir_pdf_a_markdown  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes compartidas
# ---------------------------------------------------------------------------

PALABRAS_CLAVE_DEFAULT = {
    "terminos_condiciones": [
        "términos y condiciones", "terminos y condiciones", "términos de uso",
        "condiciones de uso", "condiciones de servicio", "terms and conditions",
        "terms of use", "terms of service",
    ],
    "licencia": [
        "licencia", "licenciamiento", "licencia de uso", "license", "licensing",
        "creative commons",
    ],
    "privacidad": [
        "privacidad", "política de privacidad", "politica de privacidad",
        "aviso de privacidad", "privacy policy", "privacy notice",
    ],
    "datos_personales": [
        "protección de datos", "proteccion de datos", "tratamiento de datos",
        "política de datos", "politica de datos", "manejo de datos",
        "uso de datos", "data policy", "personal data",
    ],
    "aviso_legal": [
        "aviso legal", "notas legales", "legal notice", "disclaimer",
    ],
    "derechos_autor": [
        "derechos reservados", "derechos de autor", "derecho de autor",
        "propiedad intelectual", "copyright",
    ],
    "politicas_editoriales": [
        "política editorial", "politica editorial", "políticas editoriales",
        "politicas editoriales", "editorial policy", "editorial policies",
    ],
}

# Palabras clave para ENCONTRAR una página "Acerca de" (paso 8, último
# recurso) -- deliberadamente separadas de PALABRAS_CLAVE_DEFAULT: estas
# palabras no describen categorías legales, solo sirven para dar con el
# link a esa página. Una vez adentro, el contenido se clasifica con las
# palabras clave legales de siempre (ver _detectar_acerca_de).
PALABRAS_CLAVE_ACERCA_DE = {
    "acerca_de": [
        "acerca de", "sobre nosotros", "quiénes somos", "quienes somos",
        "sobre el sitio", "conócenos", "conocenos", "about us", "about",
    ],
}

_SELECTORES_FOOTER_DEFAULT = [
    "footer",
    "[role='contentinfo']",
    "#footer",
    ".footer",
    "[id*='footer']",
    "[class*='footer']",
]

MAX_CANDIDATOS_CRAWL_DEFAULT = 5
# Cuántos links legales anidados seguir dentro de una subpágina ya
# encontrada (ej. términos -> link a "política de privacidad completa" ->
# ese link también puede traer otro anidado) antes de cortar. 2 cubre el
# caso típico (footer -> subpágina -> lo que esa subpágina enlace) sin
# arriesgarse a crawlear el sitio entero si la estructura legal es profunda.
MAX_PROFUNDIDAD_SUBPAGINAS_DEFAULT = 2
RUTAS_WELL_KNOWN = ["/.well-known/tdmrep.json", "/.well-known/license.txt", "/license.txt", "/LICENSE"]

PAUSA = 1  # segundos entre requests, para no saturar el servidor
PAUSA_LIMITE_RITMO = 30  # segundos base de espera ante 429/503
TIMEOUT = 25
REINTENTOS = 3
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ScraperLegales/1.0)"}

RE_HF = re.compile(r"^https?://huggingface\.co/(datasets/)?([^/?#]+(?:/[^/?#]+)?)", re.IGNORECASE)
RE_GITHUB = re.compile(r"^https?://github\.com/([^/?#]+)/([^/?#]+)", re.IGNORECASE)
RE_DSPACE_ITEM = re.compile(r"^(https?://[^/]+)/items/([0-9a-fA-F-]{36})")
RE_CKAN_DATASET = re.compile(r"^(https?://[^/]+)/dataset/([^/?#]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers de texto / HTTP compartidos
# ---------------------------------------------------------------------------

def _normalizar(texto):
    """minúsculas + sin tildes, para comparar palabras clave sin depender
    de que el sitio y la lista de keywords coincidan en acentuación."""
    if not texto:
        return ""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sin_acentos.lower().strip()


def _esperar_limite_ritmo(status_code, headers, intento, contexto=""):
    prefijo = f"[{contexto}] " if contexto else ""
    espera = None
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            espera = float(retry_after)
        except ValueError:
            espera = None
    if espera is None:
        espera = PAUSA_LIMITE_RITMO * intento
    print(f"    {prefijo}status {status_code}: límite de ritmo del servidor, esperando {espera:.0f}s antes de reintentar")
    time.sleep(espera)


def _texto_decodificado(r):
    """`requests` asume ISO-8859-1 cuando el Content-Type no trae un
    charset explícito (RFC 2616), pero la gran mayoría de los sitios son
    UTF-8 con el charset declarado solo en el <meta> del HTML -- leer con
    ese default produce mojibake ("aragonÃ©s" en vez de "aragonés"). Si el
    header no trae charset, se usa el encoding detectado (chardet) en vez
    del default de `requests`."""
    if "charset" not in (r.headers.get("Content-Type") or "").lower():
        r.encoding = r.apparent_encoding
    return r.text


# Marcas típicas de sistemas anti-bot con desafío en JavaScript (Anubis,
# Cloudflare, etc.) -- `requests` no ejecuta JS, así que contra estos
# sitios nunca ve el contenido real, solo la página de desafío. Se busca
# en minúsculas sobre el HTML crudo (no hace falta _normalizar: son todas
# frases en inglés sin tildes).
_MARCAS_ANTI_BOT = [
    "checking your browser", "just a moment", "attention required",
    "ddos protection by", "anubis", "please enable javascript",
    "please turn javascript on", "verifying you are human",
    "are you a robot", "cf-browser-verification",
]


def _parece_anti_bot(html):
    """Heurística para decidir si vale la pena pagar el costo de
    reintentar con un navegador real: HTML sospechosamente corto (un
    sitio real rara vez entra en 20000 caracteres) Y con alguna marca
    típica de desafío anti-bot -- las dos condiciones juntas, para no
    activar el fallback (mucho más lento) en páginas que simplemente son
    cortas de verdad."""
    if len(html) > 20000:
        return False
    html_lower = html.lower()
    return any(marca in html_lower for marca in _MARCAS_ANTI_BOT)


def _obtener_html_stealthy(url, contexto=""):
    """Fallback para sitios protegidos por un desafío anti-bot en
    JavaScript (ej. Anubis, confirmado en repositoriodigital.uct.cl) que
    `requests` no puede resolver -- usa un navegador real vía
    `scrapling.StealthyFetcher`. Mucho más lento que requests (levanta un
    navegador entero), así que solo se llama como respaldo puntual desde
    `obtener_html_requests` cuando `_parece_anti_bot` da positivo, nunca
    como fetcher por default."""
    prefijo = f"[{contexto}] " if contexto else ""
    try:
        from scrapling import StealthyFetcher
    except ImportError:
        print(f"    {prefijo}scrapling no está instalado, no se puede usar el fallback anti-bot")
        return None
    try:
        pagina = StealthyFetcher.fetch(url, headless=True, network_idle=True, timeout=TIMEOUT * 1000)
        if pagina.status != 200:
            print(f"    {prefijo}StealthyFetcher status {pagina.status} en {url}")
            return None
        html = pagina.body
        return html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html
    except Exception as e:
        print(f"    {prefijo}error en StealthyFetcher para {url}: {e}")
        return None


def obtener_html_requests(url, contexto=""):
    """Fetch por default para páginas/subpáginas: requests liso, con
    reintentos + manejo de 429/503. Sirve para la mayoría de los sitios
    institucionales; si un sitio necesita JS/impersonación (WAF, ej.
    csjn.gov.ar), se puede inyectar una función propia vía `obtener_html`
    en las funciones de este módulo que lo aceptan.

    Además, si el HTML que devuelve un 200 tiene pinta de página de
    desafío anti-bot (`_parece_anti_bot`), reintenta esa misma URL con
    `_obtener_html_stealthy` (navegador real) antes de darla por buena --
    si el fallback también falla, se queda con el HTML del desafío (no
    tiene texto legal real, pero es mejor que reintentar en un loop
    infinito contra el mismo muro)."""
    prefijo = f"[{contexto}] " if contexto else ""
    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                html = _texto_decodificado(r)
                if _parece_anti_bot(html):
                    print(f"    {prefijo}posible muro anti-bot en {url}, reintentando con navegador")
                    html_stealthy = _obtener_html_stealthy(url, contexto=contexto)
                    if html_stealthy:
                        return html_stealthy
                return html
            if r.status_code in (429, 503):
                _esperar_limite_ritmo(r.status_code, r.headers, intento, contexto=contexto)
                continue
            print(f"    {prefijo}status {r.status_code} en {url} (intento {intento})")
        except Exception as e:
            print(f"    {prefijo}error al pedir {url}: {e} (intento {intento})")
        time.sleep(PAUSA * intento)
    return None


def _descargar_pdf(url, ruta, contexto=""):
    if ruta.exists() and ruta.stat().st_size > 0:
        return "ok"  # ya descargado en una corrida anterior
    prefijo = f"[{contexto}] " if contexto else ""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                ruta.write_bytes(r.content)
                return "ok"
            if r.status_code in (429, 503):
                _esperar_limite_ritmo(r.status_code, r.headers, intento, contexto=contexto)
                continue
            print(f"        {prefijo}status {r.status_code} al bajar {url} (intento {intento})")
        except Exception as e:
            print(f"        {prefijo}excepción al bajar {url}: {e} (intento {intento})")
        time.sleep(PAUSA * intento)
    return "error"


def _obtener_json(url, contexto="", reintentos=2):
    """GET con manejo de 429/503; cualquier otro código (404 típicamente,
    cuando el adaptador de turno no aplica a este sitio) se trata como
    "no encontrado" sin ruido -- es el caso normal, no un error."""
    prefijo = f"[{contexto}] " if contexto else ""
    for intento in range(1, reintentos + 1):
        try:
            r = requests.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=TIMEOUT)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    return None
            if r.status_code in (429, 503):
                _esperar_limite_ritmo(r.status_code, r.headers, intento, contexto=contexto)
                continue
            return None
        except Exception as e:
            print(f"    {prefijo}error al pedir {url}: {e} (intento {intento})")
        time.sleep(PAUSA * intento)
    return None


def _url_base(url):
    partes = urlsplit(url)
    return f"{partes.scheme}://{partes.netloc}"


# ---------------------------------------------------------------------------
# Detección de footer / clasificación de links y texto (compartido por
# varios pasos de detectar_licencia)
# ---------------------------------------------------------------------------

def clasificar_texto_y_href(texto, href, palabras_clave):
    texto_norm = _normalizar(texto)
    href_norm = _normalizar(href)
    for categoria, palabras in palabras_clave.items():
        for palabra in palabras:
            palabra_norm = _normalizar(palabra)
            if palabra_norm and (palabra_norm in texto_norm or palabra_norm in href_norm):
                return categoria
    return None


def encontrar_footer(soup, selector_extra=None):
    """Prueba selectores comunes de footer en orden hasta encontrar uno con
    contenido. `selector_extra` (CSS) se prueba primero, para sitios que no
    entran en ningún patrón default."""
    selectores = ([selector_extra] if selector_extra else []) + _SELECTORES_FOOTER_DEFAULT
    for selector in selectores:
        try:
            encontrados = soup.select(selector)
        except Exception:
            continue
        for tag in encontrados:
            if tag.get_text(strip=True):
                return tag
    return None


def clasificar_enlaces_footer(footer_tag, url_pagina, palabras_clave):
    """Links dentro de `footer_tag` (funciona sobre cualquier tag de bs4
    con <a href>, no solo un <footer> -- se usa también para juntar
    candidatos de la página completa en el paso de crawl) cuyo texto
    visible o href matchea alguna palabra clave, deduplicados por url."""
    candidatos = []
    vistos = set()
    for a in footer_tag.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "tel:")):
            continue
        texto = a.get_text(strip=True)
        categoria = clasificar_texto_y_href(texto, href, palabras_clave)
        if categoria is None:
            continue
        url_absoluta = urljoin(url_pagina, href)
        if url_absoluta in vistos:
            continue
        vistos.add(url_absoluta)
        candidatos.append({"categoria": categoria, "texto": texto, "href": href, "url": url_absoluta})
    return candidatos


def es_pdf(url):
    return bool(re.search(r"\.pdf(?:[?#].*)?$", url, re.IGNORECASE))


def _contenido_desde_heading(heading_tag):
    """Junta un heading con sus hermanos siguientes hasta el próximo
    heading del mismo nivel o más alto (patrón típico de "acordeón" de
    títulos + párrafos) y lo convierte a Markdown."""
    nivel = int(heading_tag.name[1])
    partes = [str(heading_tag)]
    for hermano in heading_tag.find_next_siblings():
        if hermano.name and re.match(r"^h[1-6]$", hermano.name) and int(hermano.name[1]) <= nivel:
            break
        partes.append(str(hermano))
    return markdownify.markdownify("".join(partes), heading_style="ATX").strip()


def escanear_headings(soup, palabras_clave, raiz=None):
    """Recorre TODOS los headings (h1-h6) de `raiz` (por default, todo el
    documento) buscando los que matcheen alguna palabra clave en su propio
    texto -- encuentra una sección de licencia en cualquier parte del body
    aunque nada apunte a ella con un link de ancla (ej. un README largo
    con un heading "## Licencia" en el medio). Devuelve una lista de
    {"categoria", "heading_texto", "contenido"} (contenido ya en
    Markdown)."""
    raiz = raiz or soup
    hallazgos = []
    for heading in raiz.find_all(re.compile(r"^h[1-6]$")):
        texto_heading = heading.get_text(strip=True)
        categoria = clasificar_texto_y_href(texto_heading, "", palabras_clave)
        if categoria is None:
            continue
        hallazgos.append({
            "categoria": categoria,
            "heading_texto": texto_heading,
            "contenido": _contenido_desde_heading(heading),
        })
    return hallazgos


# ---------------------------------------------------------------------------
# Descarga y conversión de PDF / subpáginas (usado por detectar_licencia
# cuando se le pasa directorio_salida, para bajar el contenido completo)
# ---------------------------------------------------------------------------

def _nombre_archivo_desde_url(url):
    nombre = Path(urlsplit(url).path).name or "documento"
    nombre = re.sub(r'[\\/*?:"<>|]', "-", nombre)
    if not nombre.lower().endswith(".pdf"):
        nombre += ".pdf"
    return nombre


def _slug_desde_url(url):
    """Incluye SIEMPRE el dominio, no solo el path -- si solo se usara el
    path, dos URLs de sitios distintos con el mismo path (o ambas con path
    vacío, ej. la home "https://loquesea.com/") caerían en la misma
    carpeta de salida y sus archivos se pisarían entre sí (confirmado
    2026-09-29: "archivos/pagina/subpaginas/" mezclaba contenido de
    creativecommons.org con el de otros sitios sin relación, porque varias
    URLs de entrada tenían path vacío)."""
    partes = urlsplit(url)
    base = f"{partes.netloc}-{partes.path}" if partes.path else partes.netloc
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    return slug or "pagina"


def procesar_pdf(url, directorio_salida, contexto=""):
    """Descarga un PDF y lo convierte a Markdown con pdf_a_texto.py, mismo
    criterio de borrado por confianza alta que el resto del repo."""
    directorio_pdfs = Path(directorio_salida) / "pdfs"
    ruta_pdf = directorio_pdfs / _nombre_archivo_desde_url(url)
    estado_descarga = _descargar_pdf(url, ruta_pdf, contexto=contexto)

    resultado = {"pdf_status": estado_descarga, "pdf_path": str(ruta_pdf) if estado_descarga == "ok" else None}
    if estado_descarga != "ok":
        resultado["markdown_status"] = "sin_pdf"
        resultado["markdown_path"] = None
        return resultado

    ruta_md = directorio_pdfs / "markdown" / (ruta_pdf.stem + ".md")
    ruta_md.parent.mkdir(parents=True, exist_ok=True)
    conversion = convertir_pdf_a_markdown(ruta_pdf, ruta_md=ruta_md, contexto=contexto)
    resultado["markdown_status"] = conversion["estado"]
    resultado["markdown_path"] = str(ruta_md) if conversion["estado"] == "ok" else None
    resultado["requiere_revision"] = conversion.get("requiere_revision", False)

    if conversion["estado"] == "ok" and not conversion.get("requiere_revision", False):
        ruta_pdf.unlink(missing_ok=True)
        resultado["pdf_status"] = "eliminado_confianza_alta"
        resultado["pdf_path"] = None

    return resultado


def procesar_subpagina(
    url, directorio_salida, obtener_html, selector_contenido, contexto="",
    palabras_clave=None, max_profundidad=MAX_PROFUNDIDAD_SUBPAGINAS_DEFAULT,
    profundidad_actual=1, visitados=None,
):
    """Pide una página enlazada, convierte su contenido principal a
    Markdown y sigue lo que encuentre adentro: PDFs (siempre, sin importar
    la profundidad -- son descargas terminales, no agregan otra página que
    recorrer) y, mientras no se llegue a `max_profundidad`, otros links
    HTML cuyo texto/href matchee alguna palabra clave (mismo criterio que
    el footer, `clasificar_texto_y_href`) -- ej. un link a "Ver Política de
    Privacidad completa" en medio del texto de los Términos y Condiciones.
    Recursivo: cada subpágina anidada puede a su vez traer más subpáginas,
    hasta `max_profundidad` niveles contados desde el primer link seguido
    (no desde el footer/hallazgo original).

    `visitados`: set de URLs ya procesadas en esta corrida (se comparte
    entre todas las llamadas recursivas, y también entre hallazgos
    distintos del mismo `detectar_licencia` vía `procesar_hallazgo_completo`)
    para no bajar la misma página dos veces ni entrar en loop si dos
    páginas legales se enlazan entre sí."""
    if visitados is None:
        visitados = set()
    visitados.add(url)

    html_subpagina = obtener_html(url, contexto)
    if html_subpagina is None:
        return {"status": "error_fetch", "markdown_path": None, "pdfs": [], "subpaginas": []}

    soup_sub = BeautifulSoup(html_subpagina, "lxml")

    contenido = None
    if selector_contenido:
        encontrados = soup_sub.select(selector_contenido)
        contenido = encontrados[0] if encontrados else None
    if contenido is None:
        contenido = soup_sub.body or soup_sub

    # Además de header/nav/footer, los sitios modernos (React/Vue, ej.
    # GitHub) dejan markup oculto en el DOM -- modales de login, menúes,
    # banners -- pensado para mostrarse recién con JS/CSS. markdownify no
    # respeta "display:none" ni "hidden", así que sin este filtro ese
    # texto termina mezclado con el contenido real. OJO: "aria-hidden"
    # queda afuera a propósito -- ese atributo oculta del lector de
    # pantalla, no visualmente (ver GitHub: el código de un archivo va en
    # líneas con aria-hidden="true" y filtrarlas borraba el contenido
    # real, ej. el texto completo de una LICENSE).
    for basura in contenido.select(
        "header, nav, footer, script, style, template, "
        "[hidden], [style*='display:none'], [style*='display: none']"
    ):
        basura.decompose()

    md = markdownify.markdownify(str(contenido), heading_style="ATX").strip()
    directorio_subpaginas = Path(directorio_salida) / "subpaginas"
    directorio_subpaginas.mkdir(parents=True, exist_ok=True)
    ruta_md = directorio_subpaginas / (_slug_desde_url(url) + ".md")
    ruta_md.write_text(md + "\n", encoding="utf-8")

    palabras_clave = palabras_clave or PALABRAS_CLAVE_DEFAULT
    pdfs = []
    subpaginas = []
    for a in soup_sub.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "tel:")):
            continue
        url_link = urljoin(url, href)

        if es_pdf(url_link):
            time.sleep(PAUSA)
            resultado_pdf = procesar_pdf(url_link, directorio_salida, contexto=f"{contexto} pdf")
            resultado_pdf["url"] = url_link
            pdfs.append(resultado_pdf)
            continue

        if profundidad_actual >= max_profundidad or url_link in visitados:
            continue
        texto_link = a.get_text(strip=True)
        categoria = clasificar_texto_y_href(texto_link, href, palabras_clave)
        if categoria is None:
            continue

        visitados.add(url_link)
        time.sleep(PAUSA)
        resultado_anidado = procesar_subpagina(
            url_link, directorio_salida, obtener_html, selector_contenido,
            contexto=f"{contexto} anidado", palabras_clave=palabras_clave,
            max_profundidad=max_profundidad, profundidad_actual=profundidad_actual + 1,
            visitados=visitados,
        )
        resultado_anidado["url"] = url_link
        resultado_anidado["categoria"] = categoria
        resultado_anidado["texto_link"] = texto_link
        subpaginas.append(resultado_anidado)

    return {"status": "ok", "markdown_path": str(ruta_md), "pdfs": pdfs, "subpaginas": subpaginas}


# ---------------------------------------------------------------------------
# Footer: paso interno de la cascada de detectar_licencia (no es un punto de
# entrada público -- el módulo siempre se usa dándole una URL a
# detectar_licencia/procesar_planilla, nunca un HTML ya bajado por otro lado)
# ---------------------------------------------------------------------------

def _fragmentos_de_texto(footer_tag):
    """Divide el texto del footer en fragmentos por bloque (cada tag de
    bs4 aporta un salto de línea con separator="\\n"), para poder marcar
    qué fragmento puntual menciona una licencia en vez de devolver el
    footer entero de una sola vez."""
    texto = footer_tag.get_text(separator="\n", strip=True)
    return [linea.strip() for linea in texto.split("\n") if linea.strip()]


def _detectar_footer(html, url_pagina, *, palabras_clave=None, selector_footer=None):
    """
    Paso "footer" de la cascada: busca el footer de `html` (ya descargado
    por detectar_licencia) y devuelve, sin pedir ninguna URL adicional,
    todas las menciones de licencia/términos/privacidad/etc. que
    encuentra ahí: "texto_plano" (fragmento del footer) o
    "enlace_referencia" (URL de un link, sin seguirla).

    Devuelve:
      {
        "footer_encontrado": bool,
        "hallazgos": [
          {"categoria": str, "tipo": "texto_plano", "valor": str},
          {"categoria": str, "tipo": "enlace_referencia", "valor": str, "texto_link": str},
          ...
        ],
      }
    """
    palabras_clave = palabras_clave or PALABRAS_CLAVE_DEFAULT
    soup = BeautifulSoup(html, "lxml")
    footer = encontrar_footer(soup, selector_extra=selector_footer)
    if footer is None:
        return {"footer_encontrado": False, "hallazgos": []}

    enlaces = clasificar_enlaces_footer(footer, url_pagina, palabras_clave)
    textos_de_enlaces = {e["texto"].strip() for e in enlaces if e["texto"]}

    hallazgos = [
        {"categoria": e["categoria"], "tipo": "enlace_referencia", "valor": e["url"], "texto_link": e["texto"]}
        for e in enlaces
    ]

    for fragmento in _fragmentos_de_texto(footer):
        if fragmento in textos_de_enlaces:
            continue  # ya reportado como enlace_referencia, no duplicar
        categoria = clasificar_texto_y_href(fragmento, "", palabras_clave)
        if categoria is not None:
            hallazgos.append({"categoria": categoria, "tipo": "texto_plano", "valor": fragmento})

    return {"footer_encontrado": True, "hallazgos": hallazgos}


# ---------------------------------------------------------------------------
# Pasos de la cascada de detectar_licencia: adaptadores de plataforma
# conocida (HuggingFace, GitHub, DSpace, CKAN), rel=license/JSON-LD,
# .well-known/, y el crawl de un nivel
# ---------------------------------------------------------------------------

def _adaptador_huggingface(url, contexto=""):
    m = RE_HF.match(url)
    if not m:
        return None
    es_dataset = bool(m.group(1))
    id_recurso = m.group(2)
    tipo_api = "datasets" if es_dataset else "models"
    api_url = f"https://huggingface.co/api/{tipo_api}/{id_recurso}"
    datos = _obtener_json(api_url, contexto=contexto)
    if datos is None:
        return None

    licencia = (datos.get("cardData") or {}).get("license")
    if not licencia:
        for tag in datos.get("tags", []):
            if isinstance(tag, str) and tag.startswith("license:"):
                licencia = tag.split(":", 1)[1]
                break
    if not licencia:
        return None
    if isinstance(licencia, list):
        # cardData.license puede traer más de una licencia (ej. dataset
        # con partes bajo licencias distintas) -- se juntan en un string
        # legible en vez de dejar el str(list) crudo ("['cc-by-4.0']").
        licencia = "; ".join(str(l) for l in licencia)
    return {
        "categoria": "licencia", "tipo": "estructurado", "valor": licencia,
        # HuggingFace no expone un archivo de licencia puntual como GitHub
        # (html_url) o CKAN (license_url) -- la referencia útil para un
        # humano es la página del propio recurso, no la URL de la API.
        "url_licencia": url,
        "plataforma": "huggingface", "api_url": api_url,
    }


def _adaptador_github(url, contexto=""):
    m = RE_GITHUB.match(url)
    if not m:
        return None
    owner = m.group(1)
    repo = re.sub(r"\.git$", "", m.group(2))
    api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
    datos = _obtener_json(api_url, contexto=contexto)
    if not datos or not datos.get("license"):
        return None
    licencia = datos["license"]
    return {
        "categoria": "licencia", "tipo": "estructurado",
        "valor": licencia.get("spdx_id") or licencia.get("name"),
        "url_licencia": datos.get("html_url"),
        "plataforma": "github", "api_url": api_url,
    }


def _dc_valor(metadata, campo):
    valores = metadata.get(campo)
    return valores[0]["value"] if valores else None


def _muestrear_dspace(url_base, contexto="", tamano_muestra=5):
    """La URL no apunta a un ítem puntual: DSpace no tiene "una" licencia
    de sitio (cada ítem trae la suya, o ninguna) -- en vez de fallar, se
    pide una muestra de ítems del buscador y se resume qué valores de
    dc.rights aparecen."""
    api_url = f"{url_base}/server/api/discover/search/objects"
    try:
        r = requests.get(api_url, params={"dsoType": "item", "size": tamano_muestra},
                          headers={**HEADERS, "Accept": "application/json"}, timeout=TIMEOUT)
    except Exception as e:
        print(f"    [{contexto}] error al muestrear DSpace {url_base}: {e}")
        return None
    if r.status_code != 200:
        return None
    try:
        objetos = r.json()["_embedded"]["searchResult"]["_embedded"]["objects"]
    except (KeyError, TypeError, ValueError):
        return None

    conteo = {}
    sin_licencia = 0
    for obj in objetos:
        item = obj.get("_embedded", {}).get("indexableObject", {})
        valor = _dc_valor(item.get("metadata", {}), "dc.rights")
        if valor:
            conteo[valor] = conteo.get(valor, 0) + 1
        else:
            sin_licencia += 1

    return {
        "categoria": "licencia", "tipo": "estructurado", "valor": None,
        "plataforma": "dspace", "api_url": r.url,
        "muestra_items": len(objetos), "licencias_encontradas": conteo, "items_sin_licencia": sin_licencia,
        "nota": "resumen de una muestra de items -- la licencia es por item, no por sitio; ver dc.rights de cada uno",
    }


def _adaptador_dspace(url, contexto=""):
    m = RE_DSPACE_ITEM.match(url)
    if m:
        base, item_id = m.group(1), m.group(2)
        api_url = f"{base}/server/api/core/items/{item_id}"
        datos = _obtener_json(api_url, contexto=contexto)
        if datos is None:
            return None
        metadata = datos.get("metadata", {})
        valor = _dc_valor(metadata, "dc.rights")
        valor_uri = _dc_valor(metadata, "dc.rights.uri")
        if not valor and not valor_uri:
            return {
                "categoria": "licencia", "tipo": "estructurado", "valor": None,
                "plataforma": "dspace", "api_url": api_url,
                "nota": "el item no tiene dc.rights/dc.rights.uri cargado",
            }
        return {
            "categoria": "licencia", "tipo": "estructurado",
            "valor": valor or valor_uri, "url_licencia": valor_uri,
            "plataforma": "dspace", "api_url": api_url,
        }

    # No es un ítem puntual: confirmar que el sitio sea DSpace antes de
    # gastar el request de muestreo (server/api es el discovery doc de
    # DSpace 7; si no existe o no es JSON de DSpace, no aplica).
    base = _url_base(url)
    datos = _obtener_json(f"{base}/server/api", contexto=contexto)
    if datos is None or "dspace" not in json.dumps(datos).lower():
        return None
    return _muestrear_dspace(base, contexto=contexto)


def _adaptador_ckan(url, contexto=""):
    m = RE_CKAN_DATASET.match(url)
    if not m:
        return None
    base, slug = m.group(1), m.group(2)
    api_url = f"{base}/api/3/action/package_show?id={slug}"
    datos = _obtener_json(api_url, contexto=contexto)
    if not datos or not datos.get("success"):
        return None
    resultado = datos.get("result", {})
    licencia = resultado.get("license_title")
    if not licencia:
        return None
    return {
        "categoria": "licencia", "tipo": "estructurado", "valor": licencia,
        "url_licencia": resultado.get("license_url"),
        "plataforma": "ckan", "api_url": api_url,
    }


def _adaptadores_plataforma(url, contexto=""):
    for fn in (_adaptador_huggingface, _adaptador_github, _adaptador_dspace, _adaptador_ckan):
        try:
            hallazgo = fn(url, contexto=contexto)
        except Exception as e:
            print(f"    [{contexto}] error en adaptador {fn.__name__}: {e}")
            hallazgo = None
        if hallazgo:
            return hallazgo
    return None


def _detectar_rel_license(soup, url_pagina):
    hallazgos = []
    for tag in soup.select("[rel~='license']"):
        href = tag.get("href")
        if href:
            hallazgos.append({
                "categoria": "licencia", "tipo": "enlace_referencia",
                "valor": urljoin(url_pagina, href), "texto_link": tag.get_text(strip=True),
            })
    return hallazgos


def _nodos_jsonld(datos):
    if isinstance(datos, dict):
        yield datos
        grafo = datos.get("@graph")
        if isinstance(grafo, list):
            for nodo in grafo:
                yield from _nodos_jsonld(nodo)
    elif isinstance(datos, list):
        for nodo in datos:
            yield from _nodos_jsonld(nodo)


def _detectar_jsonld_license(soup):
    hallazgos = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            datos = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for nodo in _nodos_jsonld(datos):
            if not isinstance(nodo, dict):
                continue
            licencia = nodo.get("license")
            if not licencia:
                continue
            valores = licencia if isinstance(licencia, list) else [licencia]
            for valor in valores:
                # schema.org permite que "license" sea un objeto (ej. un
                # CreativeWork con su propia url/name) en vez de un string
                # -- sin este chequeo, ese dict caía crudo en "valor" y
                # terminaba escrito como el repr de un dict en la columna.
                if isinstance(valor, dict):
                    url_valor = valor.get("url") or valor.get("@id")
                    if url_valor:
                        hallazgos.append({
                            "categoria": "licencia", "tipo": "enlace_referencia",
                            "valor": url_valor, "texto_link": valor.get("name") or "",
                        })
                    elif valor.get("name"):
                        hallazgos.append({"categoria": "licencia", "tipo": "estructurado", "valor": valor["name"]})
                    continue
                hallazgos.append({"categoria": "licencia", "tipo": "estructurado", "valor": valor})
    return hallazgos


def _detectar_well_known(url_base, contexto=""):
    hallazgos = []
    for ruta in RUTAS_WELL_KNOWN:
        url = url_base.rstrip("/") + ruta
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            # Muchos sitios (SPAs con ruteo client-side) devuelven 200 con
            # el mismo HTML de siempre para cualquier ruta inexistente --
            # eso da un falso positivo acá, porque ninguna de las rutas de
            # RUTAS_WELL_KNOWN es HTML en un sitio real (son JSON o texto
            # plano). Se descarta si el Content-Type es HTML.
            content_type = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and "html" not in content_type:
                texto = _texto_decodificado(r).strip()
                if texto:
                    hallazgos.append({
                        "categoria": "licencia", "tipo": "enlace_referencia", "valor": url,
                        "contenido": texto[:2000],
                    })
        except Exception:
            pass
        time.sleep(PAUSA)
    return hallazgos


def _crawl_candidatos(soup_home, url_home, obtener_html, palabras_clave, contexto="", max_candidatos=MAX_CANDIDATOS_CRAWL_DEFAULT):
    """clasificar_enlaces_footer funciona sobre cualquier tag de bs4 con
    <a href>, no solo un footer -- acá se le pasa la página COMPLETA para
    juntar candidatos de cualquier parte (nav, sidebar, body), no solo el
    footer (que ya se probó en el paso anterior)."""
    candidatos = clasificar_enlaces_footer(soup_home, url_home, palabras_clave)
    resultados = []
    for candidato in candidatos[:max_candidatos]:
        html_candidato = obtener_html(candidato["url"], contexto)
        time.sleep(PAUSA)
        if html_candidato is None:
            continue
        soup_candidato = BeautifulSoup(html_candidato, "lxml")
        headings = escanear_headings(soup_candidato, palabras_clave)
        if headings:
            for h in headings:
                resultados.append({
                    "categoria": h["categoria"], "tipo": "texto_plano",
                    "valor": h["contenido"], "url_origen": candidato["url"],
                })
        else:
            resultados.append({
                "categoria": candidato["categoria"], "tipo": "enlace_referencia",
                "valor": candidato["url"], "texto_link": candidato["texto"],
            })
    return resultados


def _detectar_acerca_de(soup_home, url_home, obtener_html, palabras_clave, contexto="", max_candidatos=2):
    """Paso 8 de la cascada (último recurso, después de crawl_candidato):
    busca una página "Acerca de" / "Sobre nosotros" / "Quiénes somos" --
    esas páginas a veces mencionan bajo qué términos se puede usar el
    contenido del sitio (ej. "este archivo es mantenido por... y su
    contenido es de dominio público") sin que el sitio tenga en ningún
    otro lado un link explícito a "licencia"/"términos".

    Dos pasadas de clasificación distintas: primero se usa
    PALABRAS_CLAVE_ACERCA_DE (no son categorías legales, solo sirven para
    encontrar el link a esa página); una vez adentro, se clasifica su
    contenido con las palabras clave legales de siempre (`palabras_clave`)
    -- primero buscando headings relevantes (mismo criterio que
    escanear_headings en el resto de la cascada), y si no hay ninguno,
    revisando si el texto completo de la página menciona alguna palabra
    clave legal en cualquier parte (no solo en un heading) antes de
    descartarla."""
    candidatos = clasificar_enlaces_footer(soup_home, url_home, PALABRAS_CLAVE_ACERCA_DE)
    resultados = []
    for candidato in candidatos[:max_candidatos]:
        html_candidato = obtener_html(candidato["url"], contexto)
        time.sleep(PAUSA)
        if html_candidato is None:
            continue
        soup_candidato = BeautifulSoup(html_candidato, "lxml")
        headings = escanear_headings(soup_candidato, palabras_clave)
        if headings:
            for h in headings:
                resultados.append({
                    "categoria": h["categoria"], "tipo": "texto_plano",
                    "valor": h["contenido"], "url_origen": candidato["url"],
                })
            continue
        texto_pagina = soup_candidato.get_text(separator=" ", strip=True)
        categoria = clasificar_texto_y_href(texto_pagina, "", palabras_clave)
        if categoria is not None:
            resultados.append({
                "categoria": categoria, "tipo": "enlace_referencia",
                "valor": candidato["url"], "texto_link": candidato["texto"],
            })
    return resultados


def _urls_de_hallazgo(hallazgo):
    """De dónde bajar el contenido completo de un hallazgo: la URL misma
    si es una referencia a un link, o `url_licencia` si es un hallazgo
    estructurado que trae ese campo (ej. el link al archivo LICENSE real
    de GitHub, o el dc.rights.uri de DSpace). Los hallazgos "texto_plano"
    ya tienen el contenido en `valor`, no hace falta bajar nada más."""
    if hallazgo["tipo"] == "enlace_referencia":
        return [hallazgo["valor"]]
    if hallazgo["tipo"] == "estructurado" and hallazgo.get("url_licencia"):
        return [hallazgo["url_licencia"]]
    return []


def procesar_hallazgo_completo(
    hallazgo, directorio_salida, obtener_html, selector_contenido_subpagina=None, contexto="",
    palabras_clave=None, max_profundidad_subpaginas=MAX_PROFUNDIDAD_SUBPAGINAS_DEFAULT, visitados=None,
):
    """Baja el contenido completo de lo que apunta `hallazgo` y lo deja en
    disco bajo `directorio_salida`: si es un PDF, lo descarga y convierte
    a Markdown (`procesar_pdf`); si es una página HTML, convierte su
    contenido principal a Markdown y sigue lo que tenga adentro
    (`procesar_subpagina`) -- PDFs siempre, y otras páginas legales
    anidadas hasta `max_profundidad_subpaginas`. Aplica igual sin importar
    de qué paso de la cascada haya salido el hallazgo (footer,
    .well-known/, rel=license, JSON-LD o crawl_candidato): todos pasan por
    acá para bajar su contenido completo.

    `visitados`: set compartido entre TODOS los hallazgos de una misma
    corrida de `detectar_licencia` (no solo dentro de una subpágina), para
    no bajar dos veces la misma URL si aparece referenciada desde más de
    un hallazgo.

    Modifica `hallazgo` in-place agregándole "contenido_completo": una
    lista de resultados (uno por URL procesada; normalmente una sola)."""
    urls = _urls_de_hallazgo(hallazgo)
    if not urls:
        return
    if visitados is None:
        visitados = set()
    hallazgo["contenido_completo"] = []
    for url_doc in urls:
        if url_doc in visitados:
            continue
        visitados.add(url_doc)
        if es_pdf(url_doc):
            resultado = procesar_pdf(url_doc, directorio_salida, contexto=contexto)
        else:
            resultado = procesar_subpagina(
                url_doc, directorio_salida, obtener_html, selector_contenido_subpagina,
                contexto=contexto, palabras_clave=palabras_clave,
                max_profundidad=max_profundidad_subpaginas, profundidad_actual=1, visitados=visitados,
            )
        resultado["url"] = url_doc
        hallazgo["contenido_completo"].append(resultado)
        time.sleep(PAUSA)


def detectar_licencia(
    url,
    *,
    obtener_html=None,
    contexto="",
    agotar_todas=False,
    max_candidatos_crawl=MAX_CANDIDATOS_CRAWL_DEFAULT,
    max_profundidad_subpaginas=MAX_PROFUNDIDAD_SUBPAGINAS_DEFAULT,
    palabras_clave=None,
    directorio_salida=None,
    selector_contenido_subpagina=None,
):
    """
    Explora automáticamente `url` en busca de su licencia, probando en
    cascada (de más barata/confiable a más costosa/heurística):
      1. Adaptador de plataforma conocida (HuggingFace, GitHub, DSpace, CKAN).
      2. .well-known/ (tdmrep.json, license.txt, /LICENSE) -- va antes de
         descargar la página porque no depende de su HTML.
      3. Descarga de la página.
      4. rel="license" / JSON-LD en la página ya descargada.
      5. Footer.
      6. Cualquier heading del body que matchee una palabra clave.
      7. Crawl de un nivel sobre candidatos fuera del footer.
      8. Página "Acerca de" / "Sobre nosotros" -- último recurso.
    Por default se detiene en la primera que encuentra algo; con
    `agotar_todas=True` prueba todas y junta los hallazgos de cada una.

    `directorio_salida`: si se pasa, además de detectar DÓNDE está la
    licencia, baja su contenido completo -- PDF convertido a Markdown, o
    página HTML convertida a Markdown, siguiendo además los PDFs y links
    legales anidados que tenga adentro (ver `procesar_subpagina`), hasta
    `max_profundidad_subpaginas` niveles (default 2: la subpágina
    encontrada + lo que ella misma enlace). Aplica sin importar de qué
    paso haya salido el hallazgo -- footer, .well-known/, rel=license,
    JSON-LD o crawl_candidato pasan todos por el mismo mecanismo. Cada
    hallazgo de tipo "enlace_referencia" (o "estructurado" con
    `url_licencia`) queda con un campo extra "contenido_completo". Si no
    se pasa (default None), la función solo detecta y devuelve la
    URL/texto de referencia, sin bajar nada a disco.

    Devuelve:
      {
        "url": url,
        "via": "api_conocida" | "well_known" | "rel_license" | "jsonld" |
               "footer" | "body_heading" | "crawl_candidato" | "acerca_de" |
               None,
        "hallazgos": [ {"categoria", "tipo", "valor", ...}, ... ],
        "error": str,  # solo si no se pudo ni descargar la página
      }

    tipo de cada hallazgo: "estructurado" (vino de una API/JSON-LD, el
    valor ya es el texto de la licencia o su SPDX id), "enlace_referencia"
    (una URL, sin seguirla) o "texto_plano" (un fragmento de texto ya
    extraído del HTML).
    """
    palabras_clave = palabras_clave or PALABRAS_CLAVE_DEFAULT
    obtener_html = obtener_html or obtener_html_requests
    resultado = {"url": url, "via": None, "hallazgos": []}

    def _agregar(hallazgos, via):
        if not hallazgos:
            return False
        for h in hallazgos:
            h["via"] = via
        resultado["hallazgos"].extend(hallazgos)
        if resultado["via"] is None:
            resultado["via"] = via
        return not agotar_todas

    def _finalizar():
        if directorio_salida is not None:
            # Un solo set de visitados para TODOS los hallazgos de esta
            # corrida (no uno por hallazgo) -- si dos hallazgos distintos
            # apuntan a la misma URL, o una subpágina anidada enlaza de
            # vuelta a algo ya bajado, no se descarga dos veces.
            visitados = {url}
            for h in resultado["hallazgos"]:
                procesar_hallazgo_completo(
                    h, directorio_salida, obtener_html, selector_contenido_subpagina,
                    contexto=contexto, palabras_clave=palabras_clave,
                    max_profundidad_subpaginas=max_profundidad_subpaginas, visitados=visitados,
                )
        return resultado

    # 1. plataforma conocida
    hallazgo_plataforma = _adaptadores_plataforma(url, contexto=contexto)
    if _agregar([hallazgo_plataforma] if hallazgo_plataforma else [], "api_conocida"):
        return _finalizar()

    # 2. .well-known/ -- no depende del HTML de la página, se prueba antes
    # de descargarla para no perdérselo si la home falla pero el archivo de
    # licencia sigue siendo accesible.
    if _agregar(_detectar_well_known(_url_base(url), contexto=contexto), "well_known"):
        return _finalizar()

    # 3. descarga de la página (necesaria para todos los pasos siguientes)
    html = obtener_html(url, contexto)
    if html is None:
        resultado["error"] = "no se pudo descargar la página"
        return resultado
    soup = BeautifulSoup(html, "lxml")

    # 4. rel=license / JSON-LD
    if _agregar(_detectar_rel_license(soup, url), "rel_license"):
        return _finalizar()
    if _agregar(_detectar_jsonld_license(soup), "jsonld"):
        return _finalizar()

    # 5. footer
    resultado_footer = _detectar_footer(html, url, palabras_clave=palabras_clave)
    if _agregar(resultado_footer["hallazgos"], "footer"):
        return _finalizar()

    # 6. cualquier heading del body
    hallazgos_headings = [
        {"categoria": h["categoria"], "tipo": "texto_plano", "valor": h["contenido"]}
        for h in escanear_headings(soup, palabras_clave)
    ]
    if _agregar(hallazgos_headings, "body_heading"):
        return _finalizar()

    # 7. crawl de un nivel sobre candidatos fuera del footer
    if _agregar(
        _crawl_candidatos(soup, url, obtener_html, palabras_clave, contexto=contexto, max_candidatos=max_candidatos_crawl),
        "crawl_candidato",
    ):
        return _finalizar()

    # 8. último recurso: página "Acerca de" / "Sobre nosotros"
    _agregar(
        _detectar_acerca_de(soup, url, obtener_html, palabras_clave, contexto=contexto),
        "acerca_de",
    )
    return _finalizar()


# ---------------------------------------------------------------------------
# procesar_planilla: correr detectar_licencia sobre una lista de URLs
# leída de un archivo (.csv/.xlsx/.md) y volcar el resultado a un .csv (o
# .xlsx)
# ---------------------------------------------------------------------------

def _detectar_columna_url(columnas):
    for candidata in columnas:
        norm = _normalizar(str(candidata))
        if "url" in norm or "link" in norm or "enlace" in norm:
            return candidata
    return None


_RE_URL_EN_CELDA = re.compile(r"https?://[^\s,;]+")


def _dividir_urls_de_celda(texto):
    """Si `texto` (el valor crudo de la celda de URL) trae más de un link
    junto -- separados por coma, punto y coma, "y", espacio... -- los
    separa para que cada uno se procese como una fila independiente. Si no
    matchea ningún link con esquema http(s), se devuelve `[texto]` tal
    cual (ej. una celda de puro texto como "Alianza Wikimedia"):
    `detectar_licencia` la intenta igual y falla de forma controlada,
    mismo comportamiento que antes de este split.

    Nota: solo reconoce links con esquema explícito (http:// o https://)
    -- un dominio suelto sin esquema (ej. "www.ejemplo.uy" mezclado en la
    misma celda con otros links que sí lo tienen) no se separa ni se
    procesa aparte, queda descartado junto con el resto del texto de
    relleno de la celda (comas, "y", etc.)."""
    urls = _RE_URL_EN_CELDA.findall(texto)
    return urls if urls else [texto]


def _leer_urls_csv(ruta, columna_url):
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        columnas = lector.fieldnames or []
        col = columna_url or _detectar_columna_url(columnas)
        if col is None:
            raise ValueError(f"no se encontró una columna de URL en {ruta} (columnas: {columnas}); pasar columna_url explícitamente")
        filas = []
        for fila in lector:
            valor = (fila.get(col) or "").strip()
            if not valor:
                continue
            for url in _dividir_urls_de_celda(valor):
                filas.append({**fila, "url": url})
        return filas


def _leer_urls_excel(ruta, columna_url, hoja=None):
    try:
        import openpyxl
    except ImportError as e:
        raise ImportError("openpyxl no está instalado en el .venv -- `pip install openpyxl` (activar el .venv primero)") from e

    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    ws = wb[hoja] if hoja else wb.active
    filas_iter = ws.iter_rows(values_only=True)
    header = [str(h) if h is not None else "" for h in next(filas_iter)]
    col = columna_url or _detectar_columna_url(header)
    if col is None:
        raise ValueError(f"no se encontró una columna de URL en {ruta} (columnas: {header}); pasar columna_url explícitamente")
    idx = header.index(col)

    filas = []
    for fila in filas_iter:
        if idx >= len(fila) or not fila[idx]:
            continue
        registro_base = {header[i]: fila[i] for i in range(len(header)) if i < len(fila)}
        valor = str(fila[idx]).strip()
        for url in _dividir_urls_de_celda(valor):
            filas.append({**registro_base, "url": url})
    return filas


_RE_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")
_RE_URL_SUELTA = re.compile(r"https?://[^\s)>\]]+")


def _leer_urls_markdown(ruta):
    texto = ruta.read_text(encoding="utf-8")
    filas = []
    vistos = set()
    for m in _RE_MD_LINK.finditer(texto):
        texto_link, url = m.group(1), m.group(2)
        if url not in vistos:
            vistos.add(url)
            filas.append({"url": url, "texto": texto_link})
    for m in _RE_URL_SUELTA.finditer(texto):
        url = m.group(0)
        if url not in vistos:
            vistos.add(url)
            filas.append({"url": url, "texto": ""})
    return filas


def leer_urls_de_archivo(ruta_entrada, columna_url=None, hoja=None):
    """Lee un archivo de entrada (.csv, .xlsx/.xls o .md) y devuelve una
    lista de dicts, uno por fila/link, con al menos la clave "url" (para
    csv/xlsx se preservan el resto de las columnas de cada fila).
    `columna_url`: nombre de la columna con la URL; si no se pasa, se
    autodetecta buscando una columna cuyo nombre contenga "url"/"link"/
    "enlace". `hoja`: nombre de la hoja a leer si el .xlsx tiene varias
    (default: la hoja activa del archivo -- casi nunca es la que se
    quiere en un excel con más de una hoja, conviene pasarla siempre que
    el archivo no sea de una sola hoja).

    csv/xlsx: si una celda trae más de un link junto (separados por coma,
    punto y coma, espacio...), se separan y cada uno se devuelve como su
    propia fila (mismas columnas del resto de la fila, repetidas) -- ver
    `_dividir_urls_de_celda`."""
    ruta_entrada = Path(ruta_entrada)
    extension = ruta_entrada.suffix.lower()
    if extension == ".csv":
        return _leer_urls_csv(ruta_entrada, columna_url)
    if extension in (".xlsx", ".xls"):
        return _leer_urls_excel(ruta_entrada, columna_url, hoja=hoja)
    if extension == ".md":
        return _leer_urls_markdown(ruta_entrada)
    raise ValueError(f"formato no soportado: {extension!r} (usar .csv, .xlsx/.xls o .md)")


_ETIQUETAS_CATEGORIA = {
    "terminos_condiciones": "Términos y condiciones (sin licencia explícita)",
    "licencia": "Licencia mencionada (ver comentarios)",
    "privacidad": "Política de privacidad (sin licencia explícita)",
    "datos_personales": "Política de datos (sin licencia explícita)",
    "aviso_legal": "Aviso legal (sin licencia explícita)",
    "politicas_editoriales": "Política editorial (sin licencia explícita)",
}


def clasificar_tipo_licencia(resultado):
    """Arma una etiqueta corta de "tipo de licencia" a partir del primer
    hallazgo del resultado de detectar_licencia (ej. "Apache-2.0",
    "Copyright / Todos los derechos reservados", "No encontrada"). Es un
    resumen best-effort -- el detalle completo va en la columna de
    comentarios/texto."""
    if resultado.get("error"):
        # Falló la descarga de la página (timeout, SSL, conexión rechazada,
        # etc.) -- distinto de "No encontrada", que significa que SÍ se
        # pudo revisar el sitio y no tenía licencia. Mezclar los dos casos
        # bajo la misma etiqueta esconde fallas de red/infra como si fueran
        # un resultado negativo genuino (ver nota en _urls_ya_procesadas:
        # estas filas no cuentan como "ya procesadas", para reintentarlas
        # solas en la próxima corrida).
        return "Error de descarga"
    hallazgos = resultado.get("hallazgos") or []
    if not hallazgos:
        return "No encontrada"
    primero = hallazgos[0]
    if primero["tipo"] == "estructurado" and primero.get("valor"):
        return str(primero["valor"])
    if primero["categoria"] == "derechos_autor":
        return "Copyright / Todos los derechos reservados"
    return _ETIQUETAS_CATEGORIA.get(primero["categoria"], "Ver comentarios")


def _archivos_de_documento(doc, nivel=1):
    """Aplana recursivamente lo que devolvió procesar_pdf/procesar_subpagina
    -- incluye las subpáginas anidadas que haya encontrado adentro (ver
    `max_profundidad_subpaginas` en detectar_licencia) -- en una lista
    plana de archivos, uno por PDF/página, cada uno con su "nivel" de
    anidamiento (1 = bajado directo del hallazgo, 2 = un link legal
    encontrado dentro de esa página, etc.)."""
    if "pdf_status" in doc:
        return [{
            "tipo_archivo": "pdf", "nivel": nivel, "url_origen": doc.get("url"),
            "pdf_status": doc.get("pdf_status"), "pdf_path": doc.get("pdf_path"),
            "markdown_status": doc.get("markdown_status"), "markdown_path": doc.get("markdown_path"),
        }]

    archivos = [{
        "tipo_archivo": "pagina", "nivel": nivel, "url_origen": doc.get("url"),
        "status": doc.get("status"), "markdown_path": doc.get("markdown_path"),
    }]
    for p in doc.get("pdfs") or []:
        archivos.append({
            "tipo_archivo": "pdf", "nivel": nivel, "url_origen": p.get("url"),
            "pdf_status": p.get("pdf_status"), "pdf_path": p.get("pdf_path"),
            "markdown_status": p.get("markdown_status"), "markdown_path": p.get("markdown_path"),
        })
    for sub in doc.get("subpaginas") or []:
        archivos.extend(_archivos_de_documento(sub, nivel=nivel + 1))
    return archivos


def unidades_de_resultado(resultado):
    """Lista de "unidades de texto" encontradas para esta URL -- una por
    cada pieza de texto DISTINTA que encontró la cascada, ya sea inline
    (`texto_plano`/`estructurado`/`contenido` bajado directo del hallazgo,
    sin archivo de por medio) o de un archivo bajado a disco (PDF o página
    convertida a Markdown, incluyendo subpáginas legales anidadas). Pensada
    para evaluar cada documento POR SEPARADO en vez de concatenar todo en
    un solo bloque (ver `compartido/evaluar_licencias_modo_server.py`) -- footer + una
    subpágina de términos + un PDF de licencia, por ejemplo, son 3 unidades
    independientes, no una sola.

    Cada unidad: {"categoria", "via", "url_origen", "markdown_path" (None
    si es texto inline) o "texto_inline" (None si viene de un archivo)}.
    Nunca trae ambos ni ninguno de los dos últimos campos a la vez -- usar
    `texto_de_unidad` para leer el texto real sin importar cuál sea."""
    unidades = []
    for h in resultado.get("hallazgos") or []:
        categoria, via = h.get("categoria"), h.get("via")
        if h["tipo"] in ("texto_plano", "estructurado") and h.get("valor"):
            unidades.append({
                "categoria": categoria, "via": via,
                "url_origen": h.get("url_origen") or resultado.get("url"),
                "markdown_path": None, "texto_inline": str(h["valor"]),
            })
        if h.get("contenido"):
            unidades.append({
                "categoria": categoria, "via": via,
                "url_origen": h["valor"] if h["tipo"] == "enlace_referencia" else resultado.get("url"),
                "markdown_path": None, "texto_inline": str(h["contenido"]),
            })
        for doc in h.get("contenido_completo") or []:
            for archivo in _archivos_de_documento(doc):
                if archivo.get("markdown_path"):
                    unidades.append({
                        "categoria": categoria, "via": via,
                        "url_origen": archivo.get("url_origen"),
                        "markdown_path": archivo["markdown_path"], "texto_inline": None,
                    })
    return unidades


def texto_de_unidad(unidad):
    """Texto real de una unidad de `unidades_de_resultado`: lee el archivo
    si es de tipo archivo, o devuelve el texto inline directo. Vacío (no
    excepción) si el archivo no existe o no se puede leer."""
    if unidad.get("markdown_path"):
        try:
            return Path(unidad["markdown_path"]).read_text(encoding="utf-8")
        except OSError:
            return ""
    return unidad.get("texto_inline") or ""


def texto_completo_de_resultado(resultado, max_chars=8000):
    """Junta el texto de TODAS las unidades (`unidades_de_resultado`) en un
    solo bloque, separadas por `---`, recortado a max_chars. Pensado para
    un resumen humano rápido (columna `comentarios_licencia` del csv) --
    para evaluar con un LLM conviene usar las unidades por separado en vez
    de este bloque único (ver `compartido/evaluar_licencias_modo_server.py`).

    Cada trozo que viene de un archivo bajado a disco se antepone con su
    ruta (`[archivo: ...]`) -- sin esto, cuando hay más de un PDF/subpágina
    el texto quedaba todo junto sin forma de saber a qué archivo
    corresponde cada parte, ni cuál abrir si el texto se recorta."""
    partes = []
    rutas_completas = []
    for unidad in unidades_de_resultado(resultado):
        texto = texto_de_unidad(unidad)
        if not texto.strip():
            continue
        if unidad.get("markdown_path"):
            partes.append(f"[archivo: {unidad['markdown_path']}]\n\n" + texto)
            rutas_completas.append(unidad["markdown_path"])
        else:
            partes.append(texto)

    if not partes and resultado.get("error"):
        return resultado["error"]

    texto = "\n\n---\n\n".join(p.strip() for p in partes if p and p.strip())
    if len(texto) > max_chars:
        if rutas_completas:
            nota = f"\n\n[...recortado -- texto completo en: {'; '.join(rutas_completas)}]"
        else:
            nota = "\n\n[...recortado...]"
        texto = texto[:max_chars].rstrip() + nota
    return texto


def metadatos_archivos_de_resultado(resultado):
    """Arma la lista PLANA de archivos que quedaron en disco para esta URL
    (PDFs y páginas convertidas a Markdown, incluyendo los que se
    encontraron anidados dentro de otra subpágina legal -- ver "nivel" en
    cada entrada), con el detalle de por qué vía se encontraron --
    pensado para relacionar cada fila del csv de salida con los archivos
    reales bajo `directorio_descargas` sin depender de parsear el texto
    (recortado o no) de `comentarios_licencia`. Vacío si `detectar_licencia`
    corrió sin `directorio_salida` (no se bajó nada) o si ningún hallazgo
    tenía algo para bajar."""
    archivos = []
    for h in resultado.get("hallazgos") or []:
        for doc in h.get("contenido_completo") or []:
            base = {"via": h.get("via"), "categoria": h.get("categoria")}
            for archivo in _archivos_de_documento(doc):
                archivos.append({**base, **archivo})
    return archivos


def _url_referencia_de_resultado(resultado):
    hallazgos = resultado.get("hallazgos") or []
    for h in hallazgos:
        if h["tipo"] == "enlace_referencia":
            return h["valor"]
        if h.get("url_licencia"):
            return h["url_licencia"]
    # Ningún hallazgo trajo una URL propia (ej. un adaptador nuevo que se
    # olvide de setear url_licencia) -- mejor apuntar a la página de
    # origen que dejar la columna vacía cuando sí se encontró algo.
    if hallazgos:
        return resultado.get("url") or ""
    return ""


CAMPOS_SALIDA = ["url", "licencia", "comentarios_licencia", "via", "url_referencia"]


def _urls_ya_procesadas(ruta_salida):
    """URLs que ya tienen fila en `ruta_salida` (csv o xlsx), para dedup.
    Vacío si el archivo no existe todavía.

    Las filas con licencia == "Error de descarga" quedan afuera a propósito:
    esa fila no es un resultado (no se pudo ni bajar la página), así que no
    cuenta como "ya procesada" -- al reanudar la corrida, esas URLs se
    vuelven a intentar solas en vez de quedar con un fallo de red/timeout
    puntual congelado para siempre."""
    if not ruta_salida.exists():
        return set()
    if ruta_salida.suffix.lower() == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(ruta_salida, read_only=True, data_only=True)
        ws = wb.active
        return {
            fila[0] for fila in ws.iter_rows(min_row=2, values_only=True)
            if fila and fila[0] and fila[1] != "Error de descarga"
        }
    with ruta_salida.open(encoding="utf-8", newline="") as f:
        return {
            fila["url"] for fila in csv.DictReader(f)
            if fila.get("url") and fila.get("licencia") != "Error de descarga"
        }


class _EscritorSalida:
    """Escribe el archivo de salida (csv o xlsx, según la extensión de
    `ruta_salida`) fila por fila -- no al final de la corrida, para poder
    seguir el avance con `tail -f` (csv) o simplemente abriendo el xlsx a
    mitad de una corrida larga.

    xlsx evita el problema típico de csv en Excel: si la configuración
    regional espera ";" como separador (común en instalaciones en
    español), un csv separado por "," se abre como una sola columna. xlsx
    no depende de eso -- cada valor queda en su propia celda siempre."""

    def __init__(self, ruta_salida, campos):
        self.ruta_salida = Path(ruta_salida)
        self.campos = campos
        self.es_xlsx = self.ruta_salida.suffix.lower() == ".xlsx"
        self.ruta_salida.parent.mkdir(parents=True, exist_ok=True)

        if self.es_xlsx:
            import openpyxl
            if self.ruta_salida.exists():
                self.wb = openpyxl.load_workbook(self.ruta_salida)
                self.ws = self.wb.active
            else:
                self.wb = openpyxl.Workbook()
                self.ws = self.wb.active
                self.ws.append(campos)
                self.wb.save(self.ruta_salida)
        else:
            escribir_encabezado = not self.ruta_salida.exists()
            self._f = self.ruta_salida.open("a", encoding="utf-8", newline="")
            self._escritor = csv.DictWriter(self._f, fieldnames=campos)
            if escribir_encabezado:
                self._escritor.writeheader()

    def escribir_fila(self, registro):
        if self.es_xlsx:
            self.ws.append([registro.get(c, "") for c in self.campos])
            self.wb.save(self.ruta_salida)
        else:
            self._escritor.writerow(registro)
            self._f.flush()

    def cerrar(self):
        if not self.es_xlsx:
            self._f.close()


class _EscritorJSONL:
    """Escribe un .jsonl append-only con el detalle de archivos descargados
    por URL (ver `metadatos_archivos_de_resultado`) -- mismo criterio que
    el resto del repo (una línea por registro, escrita apenas termina de
    procesarse, para poder hacer `tail -f`)."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.ruta.open("a", encoding="utf-8")

    def escribir(self, registro):
        self._f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        self._f.flush()

    def cerrar(self):
        self._f.close()


def procesar_planilla(
    ruta_entrada,
    ruta_salida=None,
    *,
    columna_url=None,
    hoja=None,
    directorio_descargas=None,
    ruta_jsonl=None,
    contexto="licencia_planilla",
    obtener_html=None,
    max_chars_texto=8000,
):
    """
    Lee `ruta_entrada` (.csv, .xlsx/.xls o .md) con una columna/lista de
    URLs, corre `detectar_licencia` sobre cada una (bajando además el
    contenido completo, salvo que se pase `directorio_descargas=False`) y
    escribe/continúa un archivo de salida con columnas: url, licencia,
    comentarios_licencia, via, url_referencia. En paralelo, escribe un
    .jsonl que relaciona cada url con el detalle de los archivos que se
    bajaron para armar `comentarios_licencia` (ver
    `metadatos_archivos_de_resultado`) -- pensado para no depender de
    parsear el texto de esa columna cuando se recorta.

    ruta_salida: default `<entrada>_licencias.csv` junto al archivo de
      entrada (nunca pisa el archivo de entrada). El formato lo decide la
      extensión: `.csv` (default) o `.xlsx` si se prefiere -- csv en Excel
      con configuración regional en español puede necesitar el separador
      ";" (";" en vez de ",") para no caer todo en una sola columna al
      abrir por doble clic.
    hoja: nombre de la hoja a leer si `ruta_entrada` es un .xlsx con
      varias hojas (default: la hoja activa -- conviene pasarla siempre
      que el archivo no sea de una sola hoja).
    directorio_descargas: carpeta donde bajar PDFs/páginas completas;
      default una carpeta `archivos/` junto al archivo de salida (misma
      convención que `scraping/extras/licencias/salida_licencias/archivos/`:
      csv/jsonl sueltos en la raíz de la carpeta de salida, y `archivos/`
      con una subcarpeta por url -- `_slug_desde_url(url)` -- adentro).
      Pasar `False` para NO bajar nada (solo detectar y devolver la
      referencia, más rápido).
    ruta_jsonl: archivo .jsonl de salida para la relación csv<->archivos;
      default `<ruta_salida>.jsonl` (mismo nombre que `ruta_salida`, con
      esa extensión) junto al archivo de salida.

    Reanudable: si `ruta_salida` ya existe, las URLs que ya tengan una
    fila ahí se saltean -- dedup por `url`, mismo criterio que el resto
    del repo. Escribe una fila apenas termina de procesar cada URL (no al
    final de la corrida).
    """
    ruta_entrada = Path(ruta_entrada)
    ruta_salida = Path(ruta_salida) if ruta_salida else ruta_entrada.with_name(ruta_entrada.stem + "_licencias.csv")
    ruta_jsonl = Path(ruta_jsonl) if ruta_jsonl else ruta_salida.with_suffix(".jsonl")

    if directorio_descargas is False:
        directorio_descargas = None
    elif directorio_descargas is None:
        directorio_descargas = ruta_salida.parent / "archivos"
    else:
        directorio_descargas = Path(directorio_descargas)

    filas = leer_urls_de_archivo(ruta_entrada, columna_url=columna_url, hoja=hoja)
    print(f"[{contexto}] {len(filas)} URLs leídas de {ruta_entrada}")

    vistos = _urls_ya_procesadas(ruta_salida)
    if vistos:
        print(f"[{contexto}] {len(vistos)} URLs ya procesadas en corridas anteriores, se saltean")

    escritor = _EscritorSalida(ruta_salida, CAMPOS_SALIDA)
    escritor_jsonl = _EscritorJSONL(ruta_jsonl)
    try:
        for i, fila in enumerate(filas, start=1):
            url = fila["url"]
            if url in vistos:
                continue
            print(f"[{contexto}] [{i}/{len(filas)}] {url}")

            directorio_url = (directorio_descargas / _slug_desde_url(url)) if directorio_descargas is not None else None
            try:
                resultado = detectar_licencia(
                    url, obtener_html=obtener_html, contexto=f"{contexto} {i}",
                    directorio_salida=directorio_url,
                )
                licencia = clasificar_tipo_licencia(resultado)
                comentarios = texto_completo_de_resultado(resultado, max_chars=max_chars_texto)
                via = resultado.get("via") or ""
                url_ref = _url_referencia_de_resultado(resultado)
                archivos = metadatos_archivos_de_resultado(resultado)
                unidades = unidades_de_resultado(resultado)
            except Exception as e:
                # Una URL rota (o un bug puntual en algún adaptador/conversión)
                # no puede tumbar una corrida de horas sobre cientos de URLs --
                # se deja constancia del error en la fila y se sigue con la
                # próxima, en vez de matar el proceso entero.
                print(f"    -> ERROR sin manejar: {e}")
                licencia, comentarios, via, url_ref = "Error de procesamiento", str(e), "error", ""
                archivos, unidades = [], []

            print(f"    -> via={via or 'ninguna'} licencia={licencia!r}")

            escritor.escribir_fila({
                "url": url, "licencia": licencia, "comentarios_licencia": comentarios,
                "via": via, "url_referencia": url_ref,
            })
            escritor_jsonl.escribir({
                "url": url, "licencia": licencia, "via": via, "url_referencia": url_ref,
                "archivos": archivos, "unidades": unidades,
            })
            vistos.add(url)
            time.sleep(PAUSA)
    finally:
        escritor.cerrar()
        escritor_jsonl.cerrar()

    print(f"[{contexto}] listo -> {ruta_salida} / {ruta_jsonl}")
    return ruta_salida


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Detecta la licencia de una lista de URLs (.csv/.xlsx/.md) y arma un csv con licencia + texto completo."
    )
    parser.add_argument("entrada", help="Archivo de entrada: .csv, .xlsx/.xls o .md")
    parser.add_argument("--salida", default=None, help="archivo de salida, .csv o .xlsx según la extensión (default: <entrada>_licencias.csv)")
    parser.add_argument("--columna-url", default=None, help="nombre de la columna con la URL (default: autodetectar)")
    parser.add_argument("--hoja", default=None, help="nombre de la hoja a leer si la entrada es .xlsx con varias hojas (default: la hoja activa)")
    parser.add_argument("--sin-descargas", action="store_true", help="no bajar PDFs/páginas completas, solo detectar la referencia")
    parser.add_argument("--descargas", default=None, help="carpeta donde bajar PDFs/páginas completas (default: archivos/ junto al archivo de salida)")
    parser.add_argument("--jsonl", default=None, help="archivo .jsonl con la relación url<->archivos descargados (default: <salida>.jsonl)")
    parser.add_argument("--contexto", default="licencia_planilla")
    args = parser.parse_args()
    procesar_planilla(
        args.entrada,
        args.salida,
        columna_url=args.columna_url,
        hoja=args.hoja,
        directorio_descargas=False if args.sin_descargas else args.descargas,
        ruta_jsonl=args.jsonl,
        contexto=args.contexto,
    )


if __name__ == "__main__":
    main()
