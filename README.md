# RAG sobre la Guía de tratamientos de control de acceso mediante sistemas biométricos

Consiste en preguntar en lenguaje natural a un documento extenso sin leerlo entero ni meter todo el PDF en el prompt.

## Stack

Python, uv, LlamaIndex, ChromaDB, OpenAI

## Arquitectura

Se separan `read_doc.py` y `query.py` con el fin de no regenerar embeddings en cada pregunta, evitando costes. La evaluación vive aparte, en `run_eval.py` (recuperación) y `run_metrics.py` (métricas).

El pipeline consta de:

- **Carga:** el archivo se convierte en 23 documents.
- **Chunking:** se elige un tamaño de 300 tokens (sweet spot escogido a mano — texto técnico) y overlap de 50, con el fin de evitar cortar ideas, además de un `SentenceSplitter`. Esto evita que el chunk sea ni demasiado grande tal que pudiera cortar argumentos, ni tan pequeño que introduzca ruido a las respuestas.
- **Embedding:** tanto el archivo como el prompt se embeben con el mismo modelo, tal que se encuentren en el mismo espacio dimensional de significado y, por ende, no existan discrepancias a la hora de medir sus distancias.
- **Chroma:** los nodos son almacenados en la base de datos vectorial para su persistencia, evitando gastos por la API del modelo.
- **Retrieval:** se extraen estos vectores almacenados mediante el algoritmo ANN con k=4 (los 4 vectores más cercanos al prompt lanzado, al cual se le hace un embedding). El nivel de k evita introducir embeddings que no aporten mucha información debido a su distancia, ya que introduciría una cantidad de tokens al llamar al modelo que solo provocaría gasto y ningún tipo de retorno informacional.
- **Generación:** se usa un modelo no tan potente, 4o-mini, ya que su trabajo es sintetizar los argumentos obtenidos por el algoritmo ANN, no debe razonar ni hacer alguna tarea compleja.

## Evaluación

Es necesario un golden set, un conjunto de preguntas de las cuales se tenga constancia de su posible respuesta o no, y si existe, del número de páginas donde se encuentra, con el fin de poder obtener métricas como el HitRate y el Reciprocal Ranking. El HR nos ayudará a encontrar un valor de k (nodos próximos a la respuesta en el espacio de significado) mediante un barrido con ese parametro (2, 4, 6) obteniendo en k=4 un valor HR@4 = 1 y a partir de el disminuye al disminuir k. La distribucion de RR nos ayuda para ver las posiciones maximas (primer acierto). El MRR es usado para comparar confugraciones cuando el hit rate esta saturado. 

En cuanto al ground truth, se eligen PAGINAS, no node_ids, ya que estos se regenera al hacer re-chunking y las paginas no (inicialmente se consiguieron 105 nodos, actualizado a 96, el golden set siguió valiendo)

### Resultados de recuperación

| métrica | valor |
|---|---|
| hit rate@2 | 0.917 (11/12) |
| hit rate@4 | 1.000 (12/12) |
| hit rate@6 | 1.000 (12/12) |
| MRR | 0.903 |

- Bajar a k=2 costaría la pérdida de una posible respuesta.
- k=4 es el valor elegido.
- A partir de ese nodo no se obtienen aciertos nuevos / puntuacion mas alta.

A pesar de obtener valores de 1 para algunos HR, hay que tener en cuenta que las preguntas escogidas a mano (21 en total, 12 respondibles y 9 no) son un conjunto pequeño, por lo que cada acierto tiene un gran peso y dificulta determinar el intervalo de confianza del comportamiento del sistema; si se aumenta este golden set, el comportamiento puede diferir hacia la baja. Por ello lo interesante del sistema es su comportamiento ante abstenciones de respuestas a lo que no se encuentra en el corpus (tanto una premisa falsa como datos ausentes).

### Abstención

Las puntuaciones (L2, no distancia del coseno) para los subconjuntos (respondible y no respondible) cubren los intervalos: R (0.422 mín — 0.629 máx) y NR (0.291 mín — 0.544 máx). El solape ocurre en gran parte del rango útil, por lo que es muy difícil encontrar un valor que permita establecer un umbral de abstención que separe ambas distribuciones limpiamente.

Cuando el postprocessor no encuentra nodos que puedan responder a la pregunta (score insuciente), LlamaIndex no llama al LLM, la abstencion resulta ser gratis y determinista, el usuario recibe siempre Empty Response (ningun nodo disponible), por lo que seria ideal envolverlo en un mensaje mas acorde al contexto.


**Compromiso.** Para hacer útil el comportamiento del modelo al recuperar información del documento es necesario establecer un umbral de abstención. Para ello se recorren varios valores teniendo en cuenta el siguiente trade-off: abstenciones correctas frente a respuestas perdidas.

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
- 0.30 y 0.35 dan la misma fila: la función es escalonada y solo cambia al cruzar un score observado. Entre 0.2911 y 0.3686 no hay nada. No es un error de cálculo.

El valor escogido es 0.45. Una posible opción ingenua habría sido usar 0.55, maximizando la precisión, pero esto daría un sistema que, con el fin de poder abstenerse ante todo, no respondería a un elevado porcentaje de preguntas. Esto no sería prudencia, sería un sistema que no funciona.

0.45 es el valor ideal respecto a estas metricas, debido al tipo de cambio marginal. Para ese valor de threshold, es el ultimo corte donde se compra mas de lo que se paga (2 abstenciones por 1 respuesta perdida). Tras este valor, se pasa a ratios 2:3 y 3:4, cuyas metricas no servirian para el tipo de sistema que se busca. 0.40 puede ser otra opcion genuinamente buena, pero al tratarse de un documento de un dominio donde una respuesta inventada es  riesgo de cumplimineto y una abstencion solo es una molestia, se prefiere aumentar las abstenciones por ese pequeno coste.

## Verificación
A partir del threshold elegido (0.45) se esperaba capturar 4/9 preguntas sin respuesta - abstencion al coste de 1/12 preguntas con respuesta.

Al volver a preguntar la GS16, se predijo que habria una perdida de las respondibles y fue esta (0.4222 < 0.45). Respecto a GS03 (0.5438, premisa falsa), consiguio conceder parte de la premisa falsa y refutar la otra parte (fallo - no refutarla completa)

Para GS06, GS20, GS08 y GS10, la prediccion de abstencion fue correcta (4/9)

Con esto, mediante source_nodes, se auditaban los fragmentos de los que venia la respuesta

## Limitaciones
- n = 12 contestables y 96 nodos: se mide este corpus, no el método.
- Durante la verificacion, el umbral trata igual dos fallos distintos: de las 9 preguntas sin respuesta, en 7 el objetivo es la abstencion, mientras que en las otras 2, lo correcto es refutar (problema de prompt, no de recuperacion)
- el golden set se escribió leyendo los nodos -> sesgo al alza del hit rate.
- chunk_size = 300 sigue elegido a ojo, sin medir. (optimizable)
- con el corte en 0.45 pasan 5 de 9 preguntas sin respuesta: el umbral es un
filtro barato, no la solución a la abstención.
- "Empty Response" es un literal de LlamaIndex llegando a la interfaz. --> seria deseable capturarlo y hacerlo legible
- reproducibilidad de los scores: sigma = 8e-5 por llamada, margen al corte de
125 sigmas (la tabla es ROBUSTA, no marginal); un atípico a 70 sigmas en otra
sesión, no reproducido y sin causa establecida.
## Ejecución

Añadir `OPENAI_API_KEY` en un archivo `.env`. Ejecutar primero el indexador (`read_doc.py`) y finalmente la consulta (`query.py`):

```bash
uv run --env-file .env python src/rag_ai/read_doc.py
```

```bash
uv run --env-file .env python src/rag_ai/query.py
```

Aviso: read_doc.py indexa sin borrar la colección: ejecutarlo sobre una base ya poblada duplica los nodos.

## Mejoras futuras

- Limpiar el ruido de PDF (portada, "En revisión", espacios).
- Segunda señal para la abstención (juez LLM sobre el contexto recuperado), ya que el score por sí solo no separa las distribuciones.
- Reranker (cross-encoder) sobre los nodos recuperados.
- Validador del golden set como script determinista.
- Recalibrar el umbral dentro del pipeline de evaluación en vez de fijarlo a mano.
