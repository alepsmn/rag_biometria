# read_doc.py -------------------------

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent.parent / "data"

reader = SimpleDirectoryReader(input_dir=data_dir)
documents = reader.load_data() # len = 23 (23 pags)

splitter = SentenceSplitter( # respeta hasta el final de la frase para cortar
    # 300-50=250 tokens efectivos para cada chunk
    chunk_size=300,
    chunk_overlap=50
)

# 105 nodos ~ [total tokens doc / 250 efectivos] - chunk
nodes = splitter.get_nodes_from_documents(documents)

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# prueba
# vector = embed_model.get_text_embedding(nodes[24].text[:50]) - gasta porque llama al embedding

client = chromadb.PersistentClient(path="./chroma_db")
#client.delete_collection(name="guia_biometria") #-> vacia sin borrar bd, OJO, si esta vacia peta

collection = client.get_or_create_collection(name="guia_biometria") # coleccion (bd vectorial) ~ tabla (sql)
vector_store = ChromaVectorStore(collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

print(collection.count())