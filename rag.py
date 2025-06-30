from typing import List
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import chromadb
from unstructured.partition.pdf import partition_pdf
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings("ignore")

from IPython.display import display, Markdown

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return a list of Document objects."""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        hi_res_model_name='yolox',
        infer_table_structure=True)
    
    if not elements:
        raise ValueError(f"No content found in the PDF file: {file_path}")

    docs = []
    for el in elements:
        doc = Document(
            page_content=el.text,
            metadata={
                "source": file_path,
                "type": el.metadata.category or el.category or "Unknown"
            }
        )
        docs.append(doc)

    return docs

doc = load_pdf("data/arsenal_financial_report.pdf")
# print(len(doc))  # Display the number of documents loaded

def create_vector_store(self, chunks: List[Document], collection_name: str = "chunked_documents") -> chromadb.Client:
    """
    Create a vector store from the clustered chunks.
    
    Args:
        chunks: A list of Document objects with embeddings.
        collection_name: The name of the collection in the vector store.

    Returns:
        A ChromaDB client with the created collection.
    """
    client = chromadb.Client()
    collection = client.create_collection(name=collection_name)

    for chunk in chunks:
        if "embedding" in chunk.metadata:
            collection.add(
                ids=[chunk.metadata["chunk_id"]],
                documents=[chunk.page_content],
                metadatas=[chunk.metadata],
                embeddings=[chunk.metadata["embedding"]]
            )

    return client

