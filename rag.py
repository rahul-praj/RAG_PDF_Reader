from typing import List
import asyncio
import os
import warnings

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from agentic_chunker import AgenticChunker

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from IPython.display import display, Markdown

import os

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
chunker = AgenticChunker()
chunked_docs = asyncio.run(chunker.chunk_documents(doc))
clusters = chunker.cluster_chunks(chunked_docs)
cluster_docs = chunker.generate_cluster_embeddings(clusters)

def create_vector_store(cluster_docs: list[Document], persist_directory: str = "./cluster_vectorstore"):
    """
    Create a vector store from the clustered chunks.
    
    Args:
        chunks: A list of Document objects with embeddings.
        collection_name: The name of the collection in the vector store.

    Returns:
        A ChromaDB client with the created collection.
    """
    texts = [doc.page_content for doc in cluster_docs]
    embeddings = [doc.metadata["embedding"] for doc in cluster_docs]
    metadatas = [doc.metadata for doc in cluster_docs]

    embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    vectorstore = Chroma.from_embeddings(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory,
        embedding=embedding_function
    )

    vectorstore.persist()
    return vectorstore

cluster_vector_store = create_vector_store(cluster_docs)
cluster_retriever = cluster_vector_store.as_retriever()
