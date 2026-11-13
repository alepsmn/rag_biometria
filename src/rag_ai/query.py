# query.py -----------------------

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SimilarityPostprocessor

import chromadb

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="guia_biometria")

vector_store = ChromaVectorStore(collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# retriever = index.as_retriever(similarity_top_k=4)
# nodes = retriever.retrieve("Que hay sobre el control de horarios?")

node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.45)]
llm = OpenAI(model="gpt-4o-mini")
query_engine = index.as_query_engine(llm=llm, node_postprocessors=node_postprocessors, similarity_top_k=4)

# 06(0.2911 fuera corpus), 20(0.3686 fuera corpus), 08(0.4004 fuera corpus), 10(0.4375 dato ausente)
# 16(0.4222 factual - contestable pero muerta) 03(0.5438 premisa falsa - umbral no la corta)
questions = ["¿Cómo se calcula la indemnización por daños a un trabajador?", 
            "¿Qué norma ISO regula los sensores de huella dactilar?",
            "¿Qué sanción económica corresponde a una infracción grave del RGPD?", 
            "¿Qué porcentaje de proporcionalidad para la intensidad de las medidas en operaciones biométricas se recomienda respecto a la sensibilidad de los datos?",
            "¿Cómo se mitiga el riesgo de identificar a personas sin participación activa?",
            "¿Una muestra biométrica eficaz, es incapaz de distinguir individuos, por lo que no es un dato personal?"]
for question in questions:  
    response = query_engine.query(question)
    print(response)

for source in response.source_nodes:
    print(source.score, source.node.get_content()[:200])

# verificacion alucinacion simple
response = query_engine.query("Que sancion economica concreta impone la guia" \
"por usar huellas dactilares sin base legal?")
print(response)