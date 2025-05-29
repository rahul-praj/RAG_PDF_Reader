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

    return [Document(page_content=doc.text, metadata={"source": file_path}) for doc in elements]

doc = load_pdf("data/arsenal_financial_report.pdf")
# print(len(doc))  # Display the number of documents loaded

def chunk_documents(documents: list[Document], chunk_overlap: int = 0) -> list[Document]:
    """Chunk documents into smaller pieces."""
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        model_name="all-MiniLM-L6-v2"
    )
    tokens = text_splitter.split_documents(documents)

    return [
        Document(
            page_content=token.page_content,
            metadata={
                "source": token.metadata.get("source", "unknown"),
                "chunk_index": i
            }
        ) for i, token in enumerate(tokens)
    ]

chunks = chunk_documents(doc, chunk_overlap=0)
print(len(chunks))  # Display the number of chunks created
print(chunks[455].page_content)  # Display the content of the first chunk

# def create_vector_store(chunks: list[Document], collection_name: str = "arsenal_financial_report") -> chromadb.Client:
#     """Create a vector store from the document chunks."""
#     embeddings = OllamaEmbeddings(model="llama2")
#     client = chromadb.Client()
#     collection = client.create_collection(name=collection_name)

#     for chunk in chunks:
#         collection.add(
#             documents=[chunk.page_content],
#             metadatas=[chunk.metadata],
#             ids=[str(chunk.metadata.get("chunk_index", "unknown"))],
#             embeddings=[embeddings.embed_documents([chunk.page_content])[0]]
#         )

#     return client

