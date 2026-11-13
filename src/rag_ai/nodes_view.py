
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR .parent.parent / "data" # .parents[1]

reader = SimpleDirectoryReader(input_dir=data_dir)
documents = reader.load_data() # len = 23 (23 pags)

splitter = SentenceSplitter( # respeta hasta el final de la frase para cortar
    # 300-50=250 tokens efectivos para cada chunk
    chunk_size=300,
    chunk_overlap=50
)

# 105 nodos ~ [total tokens doc / 250 efectivos] - chunk
nodes = splitter.get_nodes_from_documents(documents)

pages = defaultdict(list)
for i, node in enumerate(nodes):
    page = node.metadata.get("page_label", "Unknown")
    pages[page].append((i, node))

for page, page_nodes in pages.items():
    print(f"\n\n{'#' * 70}")
    print(f"PAGE {page} | {len(page_nodes)} NODES")
    print(f"{'#' * 70}")

    for node_id, node in page_nodes:
        print(f"\n--- NODE {node_id} ---")
        print(node.text)