from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from tqdm import tqdm
import json


pdf_file = 'EU AI ACT.pdf'


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 150,
    length_function = len,
    is_separator_regex = False,
)

def split_pdf_data_from_file(pdf_file):
    chunks_with_metadata = [] # accumlate chunk records

    pdf_elements = partition_pdf(pdf_file)
    elements = chunk_by_title(pdf_elements)

    chunk_seq_id = 0
    for element in tqdm(elements):
        if len(element.text) < 5:
            continue

        chunks = text_splitter.split_text(element.text)

        for chunk in chunks:
            chunks_with_metadata.append({
                    "text": chunk,
                    "chunk_seq_id": chunk_seq_id,
                    "page_number": element.metadata.to_dict()['page_number'],
                }
            )
            chunk_seq_id += 1


    return chunks_with_metadata


docs = split_pdf_data_from_file(pdf_file)


def save_docs_to_json(docs, filename):
    data = [doc for doc in docs]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_docs_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for doc in data:
        docs.append(Document(page_content=doc['text'],  
                       metadata = {
                        "source": "local",
                        "chunk_seq_id": doc['chunk_seq_id'],
                        "page_number": doc['page_number']
                    }))
    return docs

# Save the list of documents to a JSON file
save_docs_to_json(docs, 'eu_ai_act.json')

# Load the documents from the JSON file
# load_docs_from_json('eu_ai_act.json')