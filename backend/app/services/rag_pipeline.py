# backend/app/services/chunking_pipeline.py

import os
import warnings
from fastapi import UploadFile

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from app.services.agentic_chunker import AgenticChunker

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Initialize global variables
persist_dir = "./vectorstore"
collection_name = "document_clusters"
os.makedirs(persist_dir, exist_ok=True)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(name=collection_name)
vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name=collection_name,
    embedding_function=embedding_function
)

rag_prompt = PromptTemplate.from_template("""
You are a professional financial analyst AI assistant. Your role is to provide accurate, concise, and clear answers based solely on the information provided in the retrieved financial report context.

Instructions:
- Use only the provided context to answer the question. Do not use external knowledge or make assumptions.
- If the information needed to answer the question is not found in the context, respond with: "Not found in provided data."
- Ensure your response is structured professionally and is easy to understand for business stakeholders.

Context:
{context}

Question:
{question}

Answer:
""")

multiquery_prompt = PromptTemplate.from_template(
    "You are a helpful assistant rewriting user queries for document retrieval. Reformulate the query: {question}"
)

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temp directory and return path."""
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    return temp_path

def load_pdf(file_path: str) -> list[Document]:
    """Partition PDF and return list of Document objects."""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        hi_res_model_name='yolox',
        infer_table_structure=True
    )

    if not elements:
        raise ValueError(f"No content found in {file_path}")

    docs = []
    for el in elements:
        doc = Document(
            page_content=el.text,
            metadata={
                "source": os.path.basename(file_path),
                "type": el.category or "Unknown"
            }
        )
        docs.append(doc)

    return docs

async def chunk_and_embed(docs: list[Document]):
    """Chunk documents and add embeddings to vector store."""
    chunker = AgenticChunker()
    chunked_docs = await chunker.chunk_documents(docs)
    clusters = chunker.cluster_chunks(chunked_docs)
    cluster_docs = chunker.generate_cluster_embeddings(clusters)

    for doc in cluster_docs:
        collection.add(
            ids=[str(doc.metadata.get("cluster_id", doc.metadata.get("source")))],
            embeddings=[doc.metadata["embedding"]],
            documents=[doc.page_content]
        )

async def retrieve_and_generate(question: str):
    """Retrieve relevant docs and generate answer from LLM."""
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=multiquery_prompt,
        include_original=True
    )

    retrieved_docs = await retriever.aget_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = rag_prompt.format(context=context, question=question)
    response = llm.invoke(prompt)

    return response.content.strip()
