import chromadb, json

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent.parent / "eval" / "golden_set.json"
chromadb_dir = BASE_DIR.parent.parent / "chroma_db"

with open(data_dir, 'r', encoding='utf-8') as golden_set:
    questions = json.load(golden_set)

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

client = chromadb.PersistentClient(path=str(chromadb_dir))
collection = client.get_collection(name="guia_biometria")
vector_store = ChromaVectorStore(collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

retriever = index.as_retriever(similarity_top_k=6)

results = []
for question in questions:
    retrieved_nodes = retriever.retrieve(question["question"])
    retrieved_contexts = []

    for result in retrieved_nodes:
        retrieved_contexts.append(
            {
                "node_id": result.node.node_id,
                "text": result.node.get_content()[:300],
                "score": result.score,
                "page": result.node.metadata.get("page_label")
            }
        )

    results.append(
        {
            "question_id": question["id"],
            "category": question["category"],
            "question": question["question"],
            "answerable": question["answerable"],
            "evidence_pages": question["evidence_pages"],
            "retrieved_contexts": retrieved_contexts
        }
    )

output_path = BASE_DIR.parent.parent / "eval" / "retrieval_results.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=4)