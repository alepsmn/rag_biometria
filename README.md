# RAG sobre la Guía de tratamientos de control de presencia mediante sistemas biométricos

Consiste en preguntar en lenguaje natural a un documento extenso sin leerlo entero ni meter todo el PDF en el prompt.

## Stack

Python, uv, LlamaIndex, ChromaDB, OpenAI

## Arquitectura

Se separan `read_doc.py` y `query.py`, con el fin de no regenerar embeddings en cada pregunta, evitando costes.

El pipeline consta de:

- **Carga:** el archivo se convierte en 23 documents.
- **Chunking:** se elige un tamaño de 300 tokens y overlap de 50, con el fin de evitar cortar ideas, además de un `SentenceSplitter`. Esto evita que el chunk sea ni demasiado grande tal que pudiera cortar argumentos, ni tan pequeño que introduzca ruido a las respuestas.
- **Embedding:** tanto el archivo como el prompt se embeben con el mismo modelo, tal que se encuentren en el mismo espacio dimensional de significado y, por ende, no existan discrepancias a la hora de medir sus distancias.
- **Chroma:** los nodos son almacenados en la base de datos vectorial para su persistencia, evitando gastos por la API del modelo.
- **Retrieval:** se extraen estos vectores almacenados mediante el algoritmo ANN con k=4 (los 4 vectores más cercanos al prompt lanzado, al cual se le hace un embedding). El nivel de k evita introducir embeddings que no aporten mucha información debido a su distancia, pero tampoco descarta tantos.
- **Generación:** se usa un modelo no tan potente, 4o-mini, ya que su trabajo es sintetizar los argumentos obtenidos por el algoritmo ANN, no debe razonar ni hacer alguna tarea compleja.

## Verificación y limitaciones

Prueba adversarial: una pregunta de dominio que no existía en el documento (una multa inexistente); el modelo no inventó ninguna cifra. Trazabilidad con `source_nodes` para verificar los nodos y sus puntuaciones.

## Ejecución

Añadir `OPENAI_API_KEY` en un archivo `.env`. Ejecutar primero el indexador (`read_doc.py`) y finalmente la consulta (`query.py`):

```bash
uv run --env-file .env python read_doc.py
uv run --env-file .env python query.py
```

## Mejoras futuras

- Limpiar el ruido de PDF (portada, "En revision", espacios).
- Similarity cutoff para reforzar contra alucinaciones.
- Evaluación sistemática del retrieval (no dos pruebas a mano: un set de preguntas con respuestas esperadas).
- Tuning de `chunk_size`/`k` midiendo, no a ojo.
```
