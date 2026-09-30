# query.py -----------------------

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_store.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI


import chromadb

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="guia_biometria")

vector_store = ChromaVectorStore(collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# retriever = index.as_retriever(similarity_top_k=4)
# nodes = retriever.retrieve("Que hay sobre el control de horarios?")

llm = OpenAI(model="gpt-4o-mini")
query_engine = index.as_query_engine(llm=llm, similarity_top_k=4)

response = query_engine.query("Que hay sobre el control de horarios?")
print(response)

for source in response.source_nodes:
    print(source.score, source.node.get_content()[:200])

# verificacion alucinacion simple
response = query_engine.query("Que sancion economica concreta impone la guia" \
"por usar huellas dactilares sin base legal?")
print(response)