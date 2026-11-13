# Notas — Evaluación del RAG

Referencia de la sesión de tutoría. Conceptos + ejercicios + decisiones tomadas.
No es documentación del proyecto; es mi chuleta para defenderlo.

---

## 1. Qué se evalúa

Dos fallos distintos, no mezclarlos:

| Fallo | Causa | Lo mide |
|---|---|---|
| Recuperación | el retriever no trae el chunk correcto | hit rate, MRR |
| Generación | lo trae y el LLM responde mal | otra cosa (fuera de alcance) |

Si el golden set guarda la **respuesta en texto**, mido los dos a la vez y no sé
cuál me rompe. Además exigiría otro LLM que juzgue texto libre → otro proyecto.

**Decisión: evalúo solo retrieval.**

## 2. Ground truth = páginas

Para cada pregunta *answerable*, la verdad de referencia es **en qué página(s) del
PDF está la información que la responde**. Yo leo el documento y lo anoto.

Por qué página y no id del nodo: si cambio `chunk_size`, los ids cambian y el
golden set queda inservible. La página es estable frente al re-chunking.
(Decisión de diseño defendible en entrevista.)

## 3. Por qué un golden set de 15 preguntas vale para 23 páginas

Objeción: "si tengo que leer todo el corpus, ¿qué ventaja tiene el RAG?"

No etiqueto el corpus. Etiqueto una **muestra** para medir. Igual que los tests:
no verifico todas las entradas posibles de una función, escribo 10 casos.
El golden set es un **instrumento de medida**, no la base de conocimiento.
El sistema sigue sirviendo preguntas que yo nunca escribí.

Sin golden set, mi única evidencia es "lo probé y me pareció bien".

## 4. Hit rate@k

El retriever devuelve k nodos **ordenados**. Miro sus páginas:

- alguna coincide con la esperada → acierto (1)
- ninguna → fallo (0)

```
hit rate@k = aciertos / nº de preguntas answerable
```

Binario por pregunta. No importa si el nodo bueno salió 1º o 4º.
Depende de k → se calcula en k = 2, 4, 6.

**Error que cometí:** confundir "devolvió nodos" con "acertó".
El retriever **siempre** devuelve k nodos, incluso para preguntas basura.
Eso es el mecanismo, no un error.

## 5. MRR (Mean Reciprocal Rank)

Sí mide posición. *Reciprocal rank* = 1 / (posición del primer acierto), desde 1.

| Primer acierto en puesto | RR |
|---|---|
| 1 | 1 |
| 2 | 0.5 |
| 3 | 0.333 |
| 4 | 0.25 |
| ninguno | 0 |

```
MRR = media de los RR de todas las preguntas
```

**Por qué tener las dos métricas:** hit rate dice *si* llega la información;
MRR dice *lo arriba* que llega. Hit@6 alto con MRR bajo = lo bueno enterrado en
el puesto 5. Si bajo k para ahorrar tokens, se rompe.

## 6. Preguntas sin respuesta en el corpus

No tienen página esperada → **no se les calcula hit rate ni MRR**: no hay nada
que acertar.

Lo que se mide en ellas: el **score máximo** de los nodos devueltos.
Si los scores de las *out_of_corpus* son claramente más bajos que los de las
*answerable*, existe un `similarity_cutoff` que las separa → el sistema puede
abstenerse en vez de inventar.

Comparar las dos distribuciones es el objetivo final de `run_eval.py`.

## 7. Tipos de pregunta

| Tipo | Qué prueba | Página esperada |
|---|---|---|
| `answerable` | que el retriever encuentra lo que está | sí |
| `out_of_corpus` | que no finge saber lo que no está | no |
| `adversarial` | premisa falsa: que no acepta el marco de la pregunta | no |

Las *answerable* salen de mi lectura del documento. No se inventan.

## 8. Datos de la librería

Cada elemento de `response.source_nodes` (o de `retriever.retrieve(...)`) es un
`NodeWithScore`:

- `source.score` → el score cuelga del wrapper
- `source.node.metadata["page_label"]` → la página, dentro del nodo

**Trampa:** `page_label` es un **string** (`"7"`, no `7`). En Python
`7 == "7"` es `False`. Si el golden set lleva enteros, el hit rate sale 0 con un
retriever que funciona. Un solo tipo, consistente, en todo el flujo.

---

## Ejercicio 1 — cálculo a mano

| Pregunta | Esperada | Páginas recuperadas (puestos 1→6) |
|---|---|---|
| P1 | 7 | 3, 7, 12, 7, 1, 9 |
| P2 | 12 y 13 | 4, 9, 1, 12, 2, 8 |
| P3 | 2 | 5, 8, 3, 6, 9, 10 |

**MRR — resuelto:**
RR(P1) = 1/2 (página 7 en el puesto 2)
RR(P2) = 1/4 (página 12 en el puesto 4)
RR(P3) = 0 (la página 2 no aparece ni con k=6)
MRR = (1/2 + 1/4 + 0) / 3 = (3/4) / 3 = **0.25**
*(Fallo mío: sumé 5/8 en vez de 3/4.)*

**Hit rate — pendiente:**
- hit@2 = ?
- hit@4 = ?
- hit@6 = ?
- ¿Qué deduzco de la relación entre hit@4 y hit@6? → consecuencia sobre `top_k`.

---

## Decisiones tomadas

1. **Hit = al menos una página esperada en el top-k** (convención estándar;
   exigir todas sería recall@k). Si una respuesta está partida entre dos
   páginas, listo las dos y acepto cualquiera. Criterio al escribir el golden
   set: preferir preguntas cuya respuesta esté en un solo sitio.
2. No mezclar criterios entre preguntas sin dejarlo escrito en el esquema.
3. Ground truth por página, no por id de nodo.

## Para después (lista de espera — no tocar esta semana)

- Generación sintética de preguntas desde los chunks (para corpus grandes).
- Evaluación de la **generación**: faithfulness / LLM-as-judge.
- recall@k además de hit rate.
- Reranker después del retrieval.
- Barrer `chunk_size` / `overlap` midiendo con este mismo golden set.

## A defender el lunes

- Por qué evalúo retrieval y generación por separado.
- Por qué el ground truth es la página y no el id del nodo.
- Qué diferencia hay entre hit rate y MRR, y qué decido con cada una.
- Cómo elegí el `similarity_cutoff` y qué me costó (falsos negativos en
  answerable con score bajo).
- Por qué el retriever devuelve nodos incluso cuando la respuesta no está.

---

# Registro de sesión — 11 nov 2026

## Qué he aprendido (conceptos cerrados)

Ver secciones 1–8 arriba. Resumen de lo que **ya sé defender**:

- Evalúo **retrieval**, no generación. Ground truth = **páginas**, no ids de nodo.
- `hit rate@k` = aciertos / preguntas answerable. `@k` = "medido sobre los k primeros".
- `MRR` = media de 1/(puesto del primer acierto). Hit rate dice *si* llega; MRR dice *cuán arriba*.
- El retriever **siempre** devuelve k nodos. Devolver ≠ acertar.
- Una llamada al retriever con el k máximo; los demás k salen por rodajas.
- `k` no se minimiza: se baja hasta donde el hit rate aún no cae. MRR marca el suelo.
- El ahorro de bajar k está en **tokens del prompt al LLM**, no en la búsqueda vectorial.
- Las preguntas se redactan **con vocabulario de usuario**, no del documento.
  Parafrasear el nodo = contaminación del conjunto de evaluación → hit rate ≈ 1 siempre.
- Hit rate mide *presencia de la página correcta*, no *suficiencia del contenido*.

## Resultados del ejercicio

| | P1 | P2 | P3 | agregado |
|---|---|---|---|---|
| hit@2 | 1 | 0 | 0 | **1/3** |
| hit@4 | 1 | 1 | 0 | **2/3** |
| hit@6 | 1 | 1 | 0 | **2/3** |
| RR | 1/2 | 1/4 | 0 | **MRR = 0.25** |

@4 == @6 → los nodos 5 y 6 no aportan. Mantener `top_k=4` queda **justificado por datos**.

## Incidencias de infraestructura (resueltas)

Nada de esto era culpa de mover carpetas.

1. **`nodes_view.py` no se encontraba** → está en `src/rag_ai/`, no en `rag_ai/`.
2. **Rutas inconsistentes**: `data_dir` anclado a `__file__`, `./chroma_db` anclado al cwd.
   Si se lanza desde otro directorio, Chroma **no falla**: crea una base vacía nueva.
3. **`PermissionError` de nltk**: `uv` instala con hardlinks; `nltk` 3.10.3 se niega a abrir
   archivos con `st_nlink > 1` (CWE-59). Parcheado rompiendo los hardlinks del `nltk_cache`.
   Fijado con `[tool.uv] link-mode = "copy"` en `pyproject.toml`.
   *(nltk aparece porque `SentenceSplitter` lo usa para detectar finales de oración.)*
4. **1516 nodos de basura**: faltaba `llama-index-readers-file` / `pypdf`.
   Sin lector, `SimpleDirectoryReader` lee el PDF como texto plano y devuelve sus tripas
   binarias, en **1 documento sin `page_label`**. Resuelto con `uv add`.
   → **Lección: lo que no está en `dependencies`, un `uv sync` lo poda.**
5. **`golden_set.md` dentro de `data/`**: el reader ingiere el directorio entero, así que el
   set de evaluación se estaba indexando como corpus. Movido a `eval/`.
   Además `data/` está en `.gitignore`: ahí el golden set no estaba versionado.
6. **96 nodos ahora vs 105 en Chroma**: el `pypdf` nuevo extrae el texto con diferencias
   mínimas y eso mueve los cortes del splitter.
   → El golden set **sobrevivió porque el ground truth es la página**. Si fuera por id de
   nodo, trabajo perdido. Argumento de diseño verificado en la práctica.
7. **201 en Chroma**: `read_doc.py` **añade**, no reemplaza. Se reindexó sin borrar antes,
   así que la colección mezcla 105 nodos viejos + 96 nuevos. **Resuelto**: borrada y reindexada a 96.
   Dos fallos en la línea de borrado: la errata y el **orden** — borrar después de
   `get_or_create_collection` deja un asa a una colección ya destruida.
   Ojo a la errata: `delete_collection("guia_biometrica")` ≠ `guia_biometria`.

## Estado actual

- PDF: 23 documentos (1 por página), 96 nodos, páginas 1–23. Correcto.
- Chroma: **96 docs**, coherente con el código. Verificado.
- `pyproject.toml`: `llama-index-readers-file` y `link-mode = "copy"` declarados.
- `eval/`: carpeta creada, contiene el `golden_set.md` generado (descartado como borrador).

## Pendiente, en orden

1. Esquema del golden_set: 5 campos (`id`, `category`, `question`, `answerable`,
   `evidence_pages`), tipos cerrados, `evidence_pages` **siempre lista de strings**.
2. Separar en `category`: tema ausente del corpus vs tema presente con dato ausente.
   Solo el primero lo caza un `similarity_cutoff`.
3. 12 preguntas answerable escritas leyendo el volcado de nodos. Caja de tiempo: 45 min.
4. `run_eval.py`.

---

## Decisiones de redacción del golden set (11 nov, tarde)

**Mezcla deliberada de formas.** El set combina preguntas completas
(*"¿Cómo se mitiga el riesgo de...?"*) con sintagmas sin interrogación
(*"Requisitos exactos para que una medida supere el juicio de proporcionalidad"*).
Es intencionado: **los usuarios reales consultan de las dos formas**, y un golden
set compuesto solo de preguntas bien formadas mide un tipo de query que en
producción es minoritario.
*(Honestidad interna: surgió por descuido, se adopta como criterio tras verlo.)*

**Sin autorreferencias.** Nada de "según esta guía" o "las normas establecidas
aquí": el usuario no sabe que está consultando un documento concreto, y esas
palabras no existen en el corpus, así que solo añaden ruido al vector de la query.

**Nada de contenido duplicado.** Las conclusiones (páginas 22–23) repiten el cuerpo
del documento. Una pregunta cuya respuesta vive en dos sitios hace ambiguo el
ground truth: si anoto solo la conclusión, recuperar el cuerpo —igual de correcto—
cuenta como fallo. Regla: evitar esas preguntas, o listar todas las páginas.

## Validación del golden set — obligatoria antes de evaluar

Un ground truth mal anotado no produce un error visible: produce un hit rate bajo
que parece culpa del retriever. Se depura el sistema equivocado durante horas.

Comprobaciones mínimas, en código, antes de lanzar la evaluación:

1. El archivo **parsea** y tiene el número de entradas esperado.
2. `id` **único** y con el mismo formato.
3. `evidence_pages` es **lista de strings**, dentro del rango 1–23.
4. `answerable == False` ⟺ `evidence_pages == []`. Sin excepciones.
5. Todas las claves presentes en todas las entradas.
6. `category` dentro del conjunto cerrado
   {factual, premisa_falsa, dato_ausente, fuera_de_corpus}.

**Errores reales detectados por no tener esto:** páginas escritas como enteros en
vez de strings, una entrada sin la clave `evidence_pages`, dos `fuera_de_corpus`
con `answerable: true` y página asignada, ids duplicados y páginas tomadas de la
numeración impresa en vez de `page_label`.

---

## Punto de retomada (cierre del 11 nov, de madrugada)

**Hecho:** `golden_set.json` con 21 entradas verificadas · `run_eval.py` ejecutado ·
`eval/retrieval_results.json` con 21 registros × 6 nodos (página y score, strings
a los dos lados). Las métricas se calculan sobre ese archivo, sin tocar el API.

**Cuatro fallos que NO son mejoras, son correcciones** — con ellos los números
salen mal, no incompletos:

1. `evidence_pages` es una **lista** y la página del nodo un **string**:
   `["4"] == "4"` es `False`. Se necesita **pertenencia**, no igualdad.
   Es la causa de que todos los hit rate salieran 0.0.
2. El hit por pregunta es **binario** (0 o 1), no un conteo de nodos acertados.
   Y el hit rate es del **conjunto**: aciertos / nº de answerable (12), no /21.
   Por pregunta se guarda un booleano; el agregado se calcula **después** del bucle.
3. El RR es el del **primer** acierto: hay que cortar al encontrarlo, no sobrescribir.
   Y si no hay ninguno, `rr = 0` y **debe estar** en el diccionario (entra en la media).
4. El `max_score` se calcula para las **21**, no solo para las answerable: las
   `false` son la mitad baja de la comparación del cutoff.

**Primera tarea al retomar:** arreglar esos cuatro puntos y no interpretar ningún
número producido antes de hacerlo.

**Puede esperar:** renombrar la clave `metadata` → `page` en los resultados, el
validador del golden set, y el flag de reindexado en `read_doc.py`.

---

# Resultados de la evaluación (12 nov 2026)

Corpus: 1 documento, 23 páginas, 96 nodos. `chunk_size=300`, `overlap=50`,
`text-embedding-3-small`, Chroma con espacio **l2**. Retriever directo, sin LLM.
Golden set: 21 preguntas — 12 answerable, 9 sin respuesta en el corpus.

## Métricas de recuperación (solo las 12 answerable)

| métrica | valor |
|---|---|
| hit rate@2 | 0.917 (11/12) |
| hit rate@4 | 1.000 (12/12) |
| hit rate@6 | 1.000 (12/12) |
| MRR | 0.903 |

**Lectura de `@4 == @6`:** los nodos 5 y 6 no aportan ningún acierto nuevo.
`top_k = 4` queda justificado por datos, no por defecto. Bajar a 2 costaría una
pregunta (hit@2 = 11/12), así que 4 es el suelo.

**Por qué un hit@6 de 1.000 NO es para celebrar** — esto se dice antes de que lo
pregunten:

1. **n = 12.** Cada pregunta vale 8,3 puntos. Con doce muestras no se distingue un
   sistema del 100 % de uno del 80 %: el intervalo de confianza es enorme.
2. **96 nodos.** Encontrar el fragmento correcto entre 96 es un problema trivial
   comparado con hacerlo entre 100.000. La métrica mide este corpus, no el método.
3. **Las preguntas las escribí leyendo los nodos.** Aunque reformulé con vocabulario
   de usuario, elegí hechos que el texto expresa de forma distintiva. Un usuario real
   pregunta de forma más vaga y con menos suerte.
4. **k=6 sobre 96 nodos = 6 % del corpus.** Preguntar "¿está la respuesta en el 6 %
   superior?" es una prueba generosa.

**Conclusión honesta:** la recuperación no es el cuello de botella a esta escala.
El fallo interesante de este sistema es la **abstención**.

## Análisis del umbral de abstención

**El problema:** el retriever devuelve siempre k nodos, también cuando la respuesta
no está en el corpus. Sin una regla de corte, el LLM recibe contexto irrelevante y
responde igual → alucina. La regla candidata: **si el score del mejor nodo baja de X,
abstenerse.**

Para que esa regla funcione, los scores de las preguntas con respuesta tienen que ser
sistemáticamente más altos que los de las preguntas sin respuesta. Medido:

```
max_score, 12 answerable:
0.422  0.465  0.470  0.488  0.505  0.509  0.528  0.544  0.589  0.594  0.618  0.629

max_score, 9 sin respuesta:
0.291  0.369  0.400  0.438  0.460  0.488  0.521  0.525  0.544
```

**Las distribuciones se solapan.** El mínimo de las answerable (0.422) está por
debajo del máximo de las no answerable (0.544). La zona de solapamiento, 0.42–0.54,
cubre más de la mitad del rango útil.

**Consecuencia: no existe ningún umbral que separe los dos grupos limpiamente.**
No es un problema de calibración fina; es el resultado. **El score de similitud, por
sí solo, es una señal insuficiente para decidir la abstención en este corpus.**

### La tabla del compromiso

Como no hay corte limpio, la decisión es un intercambio y se presenta como tal.
Para cada umbral candidato X, dos cuentas:

- **Abstenciones correctas**: preguntas sin respuesta con `max_score < X`, de 9.
- **Respuestas perdidas**: preguntas answerable con `max_score < X`, de 12.
  Son falsos negativos: el sistema sabía contestar y se calla.

Se barre X sobre los valores observados y se tabula. El umbral elegido **no es el que
sale mejor en un número**, es el que corresponde al error que menos daño hace.

**Criterio para este dominio:** el corpus es una guía de protección de datos y el
destinatario es un entorno regulado. Una respuesta inventada es un riesgo de
cumplimiento; una abstención es una molestia. Por tanto se prefiere **perder
respuestas antes que inventarlas**: umbral alto.

### Lo que haría falta para separarlas de verdad

(No implementado; es la respuesta a "¿qué añadirías ahora?")

- **Segunda señal**: preguntar al LLM si el contexto recuperado responde la pregunta,
  en vez de fiarse solo de la distancia vectorial.
- **Reranker** (cross-encoder): puntúa pregunta y fragmento juntos, no por separado,
  y discrimina mucho mejor que la similitud de embeddings.
- **Espacio coseno** en Chroma, para que los scores sean interpretables.
- **Más preguntas sin respuesta**, para que la distribución baja tenga forma.

---

## Tabla del compromiso — umbral de abstención (12 nov 2026)

Regla evaluada: **el sistema contesta si `max_score >= X`, se abstiene si `max_score < X`.**
El operador se fija ANTES de contar. En esta rejilla no cambia ningún resultado
(ningún score observado cae sobre un X candidato; el más cercano es 0.400254),
pero una métrica sin operador declarado no es reproducible.

| X | Abstenciones correctas /9 | Respuestas perdidas /12 | Precisión | Recall |
|---|---|---|---|---|
| sin corte | 0/9 | 0/12 | 0.57 | 1.00 |
| 0.30 | 1/9 | 0/12 | 0.60 | 1.00 |
| 0.35 | 1/9 | 0/12 | 0.60 | 1.00 |
| 0.40 | 2/9 | 0/12 | 0.63 | 1.00 |
| 0.45 | 4/9 | 1/12 | 0.69 | 0.92 |
| 0.50 | 6/9 | 4/12 | 0.73 | 0.67 |
| 0.55 | 9/9 | 8/12 | 1.00 | 0.33 |

- Precisión = de las preguntas que contesta, cuántas tenían respuesta en el corpus.
- Recall (cobertura) = de las 12 contestables, cuántas llega a contestar.
- 0.30 y 0.35 dan la misma fila: la función es escalonada y solo cambia al cruzar
  un score observado. Entre 0.2911 y 0.3686 no hay nada. No es un error de cálculo.

### Tipo de cambio marginal (lo que decide, no el nivel absoluto)

| salto | abstenciones ganadas | respuestas pagadas | ¿compra barato? |
|---|---|---|---|
| 0 → 0.40 | +2 | +0 | gratis |
| 0.40 → 0.45 | +2 | +1 | sí (2:1) |
| 0.45 → 0.50 | +2 | +3 | no (2:3) |
| 0.50 → 0.55 | +3 | +4 | no (3:4) |

El codo está en 0.45. Hasta ahí cada abstención ganada cuesta menos de una
respuesta; a partir de ahí cuesta más. Ojo con la base de la columna: "4 por 1"
es 0.45 **frente a no poner umbral**; como incremento desde 0.40 son 2 por 1.
Etiquetar la base o la tabla engaña.

### Por qué 0.55 no es aceptable aunque tenga precisión 1.00

Con 0.55 el sistema se niega a contestar 8 de las 12 preguntas que sí podía
contestar. Dos de cada tres usuarios legítimos reciben "no lo sé" sobre
información que está en el documento. Recall 0.33.
No es prudencia: es un producto que no funciona. No tiene riesgo de cumplimiento
porque no tiene usuarios.
El argumento del dominio regulado justifica **inclinar** el umbral hacia arriba,
no llevarlo al extremo. Precisión 1.00 con recall 0.33 es el ejemplo de manual
de por qué la precisión nunca se reporta sola.

### El par que demuestra que ningún umbral sirve

```
answerable      GS04 :  0.5438232
no-answerable        :  0.5438314
```

El mejor score del grupo sin respuesta está 8 millonésimas POR ENCIMA de una
pregunta que sí tenía respuesta en el corpus. No es que no se haya encontrado el
umbral bueno: no existe. Cualquier corte que bloquee ese falso positivo bloquea
también una respuesta válida.

Formalmente ya se veía en las distribuciones (mín answerable 0.4222 < máx
no-answerable 0.5438), pero este par es la evidencia dura para contarlo.

### Conclusión para defender

1. La recuperación no es el cuello de botella a esta escala (96 nodos, 12
   preguntas): hit@4 = 1.0. El fallo interesante del sistema es la abstención.
2. El score de similitud, solo, es señal insuficiente para abstenerse en este
   corpus. Medido, no supuesto.
3. Un corte en 0.45 se defiende como primer filtro barato: captura 4 de 9
   preguntas sin respuesta y cuesta 1 de 12 válidas.
4. Pero deja pasar 5 de 9. Para cerrar la abstención hace falta una segunda
   señal, no un umbral mejor: juez LLM sobre el contexto recuperado
   ("¿responde esto a la pregunta?"), cross-encoder reranker, o espacio coseno.

PENDIENTE (lo cierra Milton): elegir X, escribirlo en el README con la frase de
justificación, y aplicarlo con SimilarityPostprocessor en el query engine.

---

## Prueba del cutoff en el query engine (12 nov 2026)

Configuración: `similarity_top_k=4` + `SimilarityPostprocessor(similarity_cutoff=0.45)`
en `node_postprocessors=[...]`. Import correcto: `llama_index.core.postprocessor`,
clase `SimilarityPostprocessor` (p minúscula en "processor").
`node_postprocessors` es una LISTA porque los postprocessors se encadenan: cada uno
recibe los nodos del anterior. El orden es significativo. Un reranker iría detrás.

### Hallazgo 1 — "Empty Response" no es el LLM absteniéndose

Cuando el postprocessor deja 0 nodos, LlamaIndex **no llama al LLM**: el sintetizador
cortocircuita y devuelve el literal `"Empty Response"`.

- A favor: la abstención es determinista y gratis. No depende del humor del modelo.
- En contra: `"Empty Response"` es un detalle de implementación filtrándose a la
  interfaz. Hay que interceptar el caso y devolver un mensaje real al usuario.

Salieron vacías GS06, GS20, GS08, GS10 (las 4/9 previstas) **y GS16**, que es la
contestable que la tabla predijo perder con el corte en 0.45 (su 1/12).
Predicción hecha sobre la métrica y verificada en ejecución: eso es lo que se cuenta.

### Hallazgo 2 — GS03 (premisa falsa): fallo parcial, y es de prompt

El umbral no la corta (0.55): el contexto recuperado es legítimo.
La premisa falsa tenía dos partes: (a) la muestra eficaz no distingue individuos,
(b) por tanto no es dato personal.
El modelo refutó (a) pero concedió (b) como condicional: "si no puede distinguir
entre individuos, no cumpliría con la función de ser un dato personal identificador".
Razonó DENTRO de la premisa en vez de rechazarla.

Conclusión: el umbral no puede resolver las premisas falsas, porque el fallo no está
en la recuperación. Hace falta instrucción explícita en el prompt: refutar premisas
que el contexto contradiga.

### Hallazgo 3 — fallo de método: reteclear el golden set

Las preguntas se escribieron a mano en query.py partiendo cadenas con `\`, y la
indentación de la línea siguiente entra dentro del string:
`"¿Qué norma ISO             regula los sensores..."`.
Eso NO es lo que hay en golden_set.json. Se detecta en los números: GS03 dio
max_score 0.5438232 en la evaluación y 0.5495836 en esta prueba, con otro nodo en
cabeza. Misma pregunta, score distinto = se está midiendo otra cosa.

Reglas que salen de aquí:
1. El golden set es **fuente única**. Nunca reteclear preguntas: leerlas del JSON y
   filtrar por id.
2. **Salida sin etiquetar no es un dato.** Imprimir el id junto a la respuesta; si hay
   que leer el código para saber qué produjo qué output, la medición no vale.

PENDIENTE: relanzar GS03 leyendo del JSON. Si su mejor nodo vuelve a 0.5438232, el
culpable era el espaciado. Si no, el índice ha cambiado desde la evaluación y las
métricas no corresponden a la base actual.

---

## Deriva de scores entre corridas (12 nov 2026) — CORRIGE una nota anterior

### El hecho

GS03, misma pregunta (verificado carácter a carácter contra golden_set.json),
mismo nodo en cabeza (node_id 5fd57b34-7f3d-41d2-b163-4bb2e13cbd40), misma
colección (96 docs, los 6 node_ids de la evaluación siguen existiendo → el índice
NO se ha reconstruido):

```
en la evaluacion : 0.5438314
al relanzar      : 0.5495882   (cadena limpia, leida del JSON)
                   0.5495836   (cadena con los 13 espacios de la continuacion)
```

Deriva = 0.0058. El espaciado aporta 0.000005: era irrelevante. Lo que se movió
es otra cosa, y como el índice no cambió, apunta al **vector de la consulta**:
la API de embeddings no devuelve el mismo vector para la misma cadena en días
distintos.

Medido sin API: `collection.metadata is None` → espacio **l2 por defecto**,
confirmado. Vector de nodo con norma 1.000358, dim 1536 → normalizado, así que
`score = 1 - d²` se sostiene.

### CORRECCIÓN: el par de las "8 millonésimas" NO se usa

Nota anterior decía que GS03 (0.5438314) estaba 8e-6 por encima de GS04
(0.5438232) y que eso era la evidencia dura del solapamiento. **Retirado:** era un
artefacto de una corrida. Si lo cuentas y te piden reproducirlo, no sale.

Lo que SÍ aguanta: el solapamiento. GS03 sigue por encima de preguntas
contestables y el mínimo contestable (0.4222) sigue por debajo del máximo no
contestable. La conclusión "el score solo no separa los grupos" es robusta.
La anécdota numérica no lo era.

### La consecuencia real — va al README como limitación

Un umbral fijo de 0.45 está calibrado sobre números que se mueven 0.006.
Márgenes al corte:

| pregunta | score | distancia a 0.45 |
|---|---|---|
| GS21 (sin respuesta, pasa) | 0.4600 | +0.010 |
| GS10 (sin respuesta, filtra) | 0.4375 | -0.0125 |
| GS02 (contestable, pasa) | 0.4648 | +0.015 |

El caso más cercano está a 1.7 veces la deriva observada. La tabla sobrevive,
pero con margen fino.

Frase para la entrevista: "el umbral está calibrado sobre scores que no son
bit-reproducibles entre llamadas a la API de embeddings; observé una deriva de
0.006 en el mismo par pregunta-nodo. En producción lo recalibraría como parte del
pipeline de evaluación en vez de fijarlo a mano, y guardaría los embeddings de
consulta para poder reproducir una corrida."

### Experimento pendiente (asignado)

1. Embeber la cadena de GS03 dos veces en la MISMA ejecución y comparar vectores.
   Si son idénticos → estable dentro de una corrida, deriva entre corridas.
2. Sacar el vector del nodo y calcular el score a mano: `1 - sum((q-v)**2)`.
   Si da 0.5495882 → el nodo no cambió; se movió la consulta.

Sintaxis: `collection.get(ids=[nid], include=['embeddings'])['embeddings'][0]`
y `embed_model.get_query_embedding(texto)`.

### Tercera muestra — es ruido por llamada, no deriva

```
evaluacion : 0.5438314
corrida 2  : 0.5495882   (+0.0058)
corrida 3  : 0.5439762   (+0.00014 respecto a la evaluacion)
```

No avanza en una direccion: la corrida 3 vuelve casi al valor original y la 2 fue
una excursion. Ademas, cadena sucia (0.5495836) y limpia (0.5495882) salieron casi
iguales EN LA MISMA SESION, mientras que entre sesiones varia mas. Patron: los
valores se agrupan dentro de una ventana temporal, no entre ventanas.

Con el indice intacto y la distancia calculada exactamente contra vectores float32
almacenados, el unico sitio donde puede variar es el vector de la consulta.

**Lo que se puede afirmar con 3 muestras:** "el score del mismo par pregunta-nodo
varia entre corridas hasta 0.006 y el indice no cambio".
**Lo que NO:** cual es la distribucion del ruido, ni cada cuanto ocurre una
excursion. Tres puntos no son una distribucion (misma critica que al max_score del
grupo no contestable).

### CORRIGE la nota de arriba sobre los margenes

Decia "la tabla sobrevive pero por poco". Afirmacion correcta: **ningun caso cambia
de lado**. El mas proximo al corte (GS21, +0.010) necesitaria 1.7x el ruido maximo
observado para cruzarlo. El 4/9 y el 1/12 son estables bajo este ruido. El margen es
pequeno pero suficiente.

### Medicion que cierra esto (10 llamadas, fraccion de centimo)

Embeber la misma cadena 10 veces en UNA ejecucion y mirar la dispersion:
- todos identicos -> el ruido es entre peticiones separadas en el tiempo
- varian -> es por llamada

Con eso se pasa de "vi tres numeros" a "medi la dispersion: sigma = X", que es lo
unico defendible.

### MEDIDO: dispersion intra-ejecucion (10 llamadas, GS03, mismo nodo)

```
n = 5 muestras recogidas, 4 valores distintos
0.5440082457525597   <- repetido bit a bit con el 5o
0.5438314427265831   <- EL VALOR EXACTO DE LA EVALUACION (16 digitos)
0.5439950462098058
0.5440258582870351
0.5440082457525597

rango = 0.00019
sigma = 8.0e-5
```

**Conclusion 1:** la corrida de la evaluacion NO fue anomala. Su valor aparece
exactamente en la muestra: es el regimen normal.

**Conclusion 2 — CORRIGE la nota anterior:** la nota de arriba decia "es ruido por
llamada, hasta 0.006". FALSO. El ruido por llamada es 8e-5. Los 0.006 de la
corrida 2 (0.5495882) estan a **70 sigmas** de esta media: es UN valor atipico sin
causa establecida, no la amplitud del ruido. Se confundio una observacion con una
distribucion.

**Hipotesis no confirmada:** 4 valores distintos en 5 llamadas con una repeticion
exacta sugiere un conjunto pequeno de valores discretos, tipico del orden de
reduccion en coma flotante segun el worker que atiende la peticion. Con 5 muestras
es hipotesis, NO conclusion. No contarla como tal.

### Margen al umbral, recalculado

Margen mas estrecho al corte de 0.45: GS21 a +0.010 = **125 sigmas** del ruido
intra-sesion. El corte es robusto frente al ruido medido, no "marginal".
Frente al regimen del valor atipico (0.006) el margen seria 1.7x, pero ese regimen
no esta caracterizado.

### Frase para la entrevista (esta es la buena)

"Medi la reproducibilidad de los scores: sigma = 8e-5 sobre el score del mismo par
pregunta-nodo, 125 veces menor que mi margen mas estrecho al umbral. El corte en
0.45 es estable frente a ese ruido. Observe ademas un valor atipico a 70 sigmas en
otra sesion que no he reproducido ni he podido explicar; lo senalo en vez de
esconderlo."

### Comprobacion gratis (1 minuto, manana)

Relanzar el mismo script en otra sesion/dia:
- ~0.5440 -> la corrida 2 fue una rareza irrepetible
- ~0.5496 -> hay dos regimenes y el umbral tiene riesgo real entre sesiones
