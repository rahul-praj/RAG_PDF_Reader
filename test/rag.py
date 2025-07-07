from typing import List
import asyncio
import os
import warnings

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from backend.app.services.agentic_chunker import AgenticChunker

warnings.filterwarnings("ignore")
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
                "type": el.category or "Unknown"
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
    Create a vector store from cluster-level Document objects with precomputed embeddings.

    Args:
        cluster_docs: A list of Document objects, each representing a cluster, with embeddings in metadata.
        persist_directory: Directory path to persist the Chroma vector store.

    Returns:
        A ChromaDB collection containing the cluster documents.
    """

    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="financial_clusters")

    for doc in cluster_docs:
        # Remove embeddings from metadata to avoid duplication
        # metadata_without_embedding = {k: v for k, v in doc.metadata.items() if k != "embedding"}

        collection.add(
            ids=[str(doc.metadata["cluster_id"])],
            embeddings=[doc.metadata["embedding"]],
            documents=[doc.page_content]
        )

    return collection

def create_cluster_retriever(persist_directory: str = "./cluster_vectorstore", collection_name: str = "financial_clusters"):
    """
    Create a LangChain retriever from an existing ChromaDB collection of clusters.

    Args:
        persist_directory: Directory path where the ChromaDB vector store is persisted.
        collection_name: Name of the collection containing cluster-level documents.

    Returns:
        A LangChain retriever object.
    """

    # Initialize embedding function
    embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Load existing vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    retriever = vectorstore.as_retriever()

    return retriever

# Create ChromaDB collection
collection = create_vector_store(cluster_docs)

# Create retriever from the persisted collection
cluster_retriever = create_cluster_retriever()

# Use multi-query retriever for better performance
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
retriever = MultiQueryRetriever.from_llm(
    retriever=cluster_retriever,
    llm=llm,
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant rewriting user queries for document retrieval. Reformulate the query: {question}"
    ),
    include_original=True
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

async def rag_query(question: str, retriever, model: ChatOpenAI):
    """
    Perform a RAG query using the provided question and retriever.
    
    Args:
        question (str): The question to ask.
        retriever (MultiQueryRetriever): The retriever to use for fetching relevant documents.
        model (ChatOpenAI): The language model to generate the answer.

    Returns:
        str: The generated answer.
    """
    # Retrieve relevant documents based on the question
    retrieved_docs = await retriever.aget_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = rag_prompt.format(context=context, question=question)
    response = model.invoke(prompt)
    return response.content.strip()

async def main():
    # Initialize the language model
    model = ChatOpenAI(model_name="gpt-4", temperature=0.2)

    # Example question
    question = "What were the key financial highlights of Arsenal's 2023/24 season?"

    # Perform the RAG query
    answer = await rag_query(question, retriever, model)
    
    # Display the answer
    print(f"\n **Question:** {question}\n\n **Answer:** {answer}")

if __name__ == "__main__":
    asyncio.run(main())
