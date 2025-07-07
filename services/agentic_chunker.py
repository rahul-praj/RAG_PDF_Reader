from typing import List, Union
import numpy as np
import random
import os
import json
import hashlib
from dotenv import load_dotenv


from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer

from unstructured.partition.pdf import partition_pdf
import uuid

import openai
import asyncio

import hdbscan
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt


# Load environment variables from .env file
load_dotenv()

# Define sentence transformer model for chunk summarization
sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

class AgenticChunker:

    PROMPT = ChatPromptTemplate.from_messages([

        ("system",
            """
        You are a helpful assistant that summarizes chunks of text from a specific document.
        Your task is to read the content of each chunk and generate a concise summary that captures the main points.
        The summary should be informative and relevant to the content of the chunk.

        Example:
        Chunk: The club reported a record increase in ticket sales and hospitality revenue, attributed to a strong home fixture calendar.
        Summary: Ticket and hospitality revenue rose due to a strong home match calendar.


        Another example:
        Chunk: The company’s net income fell 8% due to increased operational expenses and foreign exchange losses.
        Summary:
        Net income declined 8% due to higher operational costs and FX losses.


        Only respond with the chunk summary, nothing else.
            """
            ),
        ("user", "Chunk:\n{chunk}")
        ])

    def __init__(self, openai_api_key: Union[str, None] = None, model_name: str = "all-mpnet-base-v2"):
        """Initialize the AgenticChunker with an OpenAI API key.
        Args:
            openai_api_key: Optional; if not provided, it will look for the OPENAI_API_KEY environment variable.
        """

        self.id_length_lim = 6  # Limit for truncating UUIDs to keep chunk IDs short

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)

        self.model = sentence_transformer_model

    # Need a function that applies the other functions to ultimately generate a final set of embeddings for the chunk summaries
    async def get_embeddings(self, documents: List[Document], chunk_overlap: int = 0) -> List[Document]:
        """
        Process a list of Document objects by chunking, summarizing, and clustering them.
        
        Args:
            documents: A list of Document objects to be processed.
            chunk_overlap: The number of overlapping tokens between chunks.

        Returns:
            A list of Document objects with embeddings generated for each chunk summary.
        """
        chunked_docs = await self.chunk_documents(documents, chunk_overlap)
        clusters = self.cluster_chunks(chunked_docs)
        clustered_docs = [doc for cluster in clusters for doc in cluster]
        return clustered_docs
    
    def hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
    
    async def generate_summaries_batch_with_cache(self, tokens: List[Document], batch_size: int = 5, cache_file="summary_cache.json") -> List[str]:

        """
        Batch summarise tokens with caching.
        """
        summaries = []

        # Load cache
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}

        # Break into batches
        batches = [tokens[i:i+batch_size] for i in range(0, len(tokens), batch_size)]

        for batch in batches:
            # Check cache for each token
            uncached_indices = []
            uncached_tokens = []
            batch_summaries = [None] * len(batch)

            for idx, token in enumerate(batch):
                h = self.hash_text(token.page_content)
                if h in cache:
                    batch_summaries[idx] = cache[h]
                else:
                    uncached_indices.append(idx)
                    uncached_tokens.append(token)

            # If any tokens are uncached, summarise them
            if uncached_tokens:
                prompt_text = "\n\n".join([f"Chunk {i+1}:\n{token.page_content}" for i, token in enumerate(uncached_tokens)])
                prompt = f"""
                Summarise each chunk concisely. Output only the summaries in order, formatted as:

                Summary 1: ...
                Summary 2: ...
                ...

                Chunks:
                {prompt_text}
                """

                chain = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant summarising text chunks."),
                    ("user", prompt)
                ]) | self.llm

                output = await chain.ainvoke({})

                parsed_summaries = [line.split(": ", 1)[1] for line in output.content.strip().split("\n") if line.startswith("Summary")]

                # Update batch_summaries and cache
                for idx, summary, token in zip(uncached_indices, parsed_summaries, uncached_tokens):
                    batch_summaries[idx] = summary
                    cache[self.hash_text(token.page_content)] = summary

            summaries.extend(batch_summaries)

        # Save updated cache
        with open(cache_file, "w") as f:
            json.dump(cache, f)

        return summaries

    # Need a function that loops through each chunk and adds metadata to each Document object, including chunk ID, index, and summary.
    async def chunk_documents(self, documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 10) -> List[Document]:
        """
        Chunk non-table documents into smaller pieces and generate summaries.
        Preserve table documents as-is and skip summarization.

        Args:
            documents: List of Document objects with metadata from Unstructured.
            chunk_overlap: Number of overlapping tokens between chunks.

        Returns:
            List of Document objects with chunked content and metadata (including summaries).
        """

        # Separate table and non-table documents
        table_docs = [doc for doc in documents if "table" in doc.metadata.get("type", "").lower()]
        text_docs = [doc for doc in documents if "table" not in doc.metadata.get("type", "").lower()]

        # Apply token splitter only to non-table text documents
        text_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
            model_name="all-mpnet-base-v2"
        )
        tokens = text_splitter.split_documents(text_docs)

        chunked_docs = []

        # Concurrency and backoff

        semaphore = asyncio.Semaphore(3)  # limit to 3 concurrent requests

        # Process non-table chunks with LLM summarization
        summaries = await self.generate_summaries_batch_with_cache(tokens, batch_size=5)

        for i, (token, summary) in enumerate(zip(tokens, summaries)):
            chunk_id = str(uuid.uuid4())[:self.id_length_lim]
            chunked_docs.append(Document(
                page_content=token.page_content,
                metadata={
                    "source": token.metadata.get("source", "unknown"),
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "chunk/proposition": token.page_content,
                    "summary": summary,
                    "type": token.metadata.get("type", "text")
                }
            ))

        # Process table chunks — no splitting or summarization
        for i, table_doc in enumerate(table_docs):
            chunk_id = str(uuid.uuid4())[:self.id_length_lim]
            chunked_docs.append(Document(
                page_content=table_doc.page_content,
                metadata={
                    "source": table_doc.metadata.get("source", "unknown"),
                    "chunk_index": len(chunked_docs) + i,
                    "chunk_id": chunk_id,
                    "chunk/proposition": table_doc.page_content,
                    "summary": table_doc.page_content,  # Table content preserved
                    "type": table_doc.metadata.get("type", "table")
                }
            ))

        return chunked_docs
    
    def generate_embeddings(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        summaries = [doc.metadata.get("summary", "") for doc in docs]
        embeddings = self.model.encode(summaries, convert_to_tensor=False)

        # Add embeddings to each Document's metadata
        for doc, embedding in zip(docs, embeddings):
            doc.metadata["embedding"] = embedding.tolist()

        return docs

    
    # Need a function that creates a clustering algorithm that uses the embeddings of the chunk summaries to group similar chunks together.
    def cluster_chunks(self, chunks: List[Document]) -> List[List[Document]]:
        """
        Cluster chunks based on their summaries using embeddings.
        
        Args:
            chunks: A list of Document objects with summaries.

        Returns:
            A list of clusters, where each cluster is a list of Document objects.
        """
        enriched_docs = self.generate_embeddings(chunks)
        embeddings = np.array([doc.metadata["embedding"] for doc in enriched_docs])

        if len(enriched_docs) == 0:
            return []

        clustering_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', core_dist_n_jobs=-1)

        # Implement PCA to reduce dimensionality before clustering
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        pca_embeddings = pca.fit_transform(embeddings)
        labels = clustering_model.fit_predict(pca_embeddings)

        clusters = {}

        for label, doc in zip(labels, enriched_docs):
            doc.metadata["cluster"] = int(label)
            if label == -1:
                continue
            clusters.setdefault(label, []).append(doc)

        return list(clusters.values())
    
    def generate_cluster_embeddings(self, clusters: List[List[Document]]) -> List[Document]:
        """
        Generate representative embeddings for each cluster by averaging member embeddings.
        """
        cluster_docs = []

        for cluster in clusters:
            cluster_id = cluster[0].metadata["cluster"]
            source = cluster[0].metadata.get("source", "unknown")
            combined_summary = " ".join([doc.metadata.get("summary", "") for doc in cluster])
            embeddings = np.array([doc.metadata["embedding"] for doc in cluster])
            avg_embedding = np.mean(embeddings, axis=0)

            cluster_doc = Document(
                page_content=combined_summary,
                metadata={
                    "cluster_id": int(cluster_id),
                    "source": source,
                    "embedding": avg_embedding.tolist(),
                    "num_chunks": len(cluster)
                }
            )
            cluster_docs.append(cluster_doc)

        return cluster_docs
    
    # Need a function that projects the clustered chunks into a visualization space using UMAP
    @staticmethod
    def visualize_clusters_with_umap(embeddings: np.ndarray, labels: List[int], chunks: List[Document]):
        """
        Visualize clustered embeddings in 2D space using UMAP.

        Args:
            embeddings: A 2D NumPy array of shape (n_chunks, embedding_dim)
            labels: A list of cluster labels (one per chunk)
            chunks: The original Document chunks (used for optional annotations)
        """
        if embeddings.shape[0] == 0:
            print("No embeddings to visualize.")
            return

        # Reduce to 2D with UMAP
        reducer = UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

        # Handle colors: assign a color per cluster label (noise = -1)
        unique_labels = sorted(set(labels))
        palette = plt.cm.get_cmap('tab10', len(unique_labels))
        colors = [palette(unique_labels.index(label)) for label in labels]

        # Plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=colors, s=60, alpha=0.8, edgecolors='k'
        )

        plt.title("UMAP Projection of Clustered Chunks")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)

        # annotate noise
        if -1 in labels:
            noise_indices = [i for i, l in enumerate(labels) if l == -1]
            plt.scatter(
                embedding_2d[noise_indices, 0],
                embedding_2d[noise_indices, 1],
                c='lightgray', label='Noise', s=60, edgecolors='k'
            )

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
                            markerfacecolor=palette(unique_labels.index(label)), markersize=8)
                for label in unique_labels if label != -1]
        if -1 in unique_labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Noise',
                                    markerfacecolor='lightgray', markersize=8))
        plt.legend(handles=handles, loc='best')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    chunker = AgenticChunker()

    # Sample documents
    """Load a PDF file and return a list of Document objects."""
    elements = partition_pdf(
        filename="data/arsenal_financial_report.pdf",
        strategy="hi_res",
        hi_res_model_name='yolox',
        infer_table_structure=True)

    docs = []
    for el in elements:
        doc = Document(
            page_content=el.text,
            metadata={
                "source": "data/arsenal_financial_report.pdf",
                "type": el.category or "Unknown"
            }
        )
        docs.append(doc)

    print(f"Loaded {len(docs)} documents from the PDF.")

    processed_docs = asyncio.run(chunker.get_embeddings(docs, chunk_overlap=10))
    embeddings = np.array([doc.metadata["embedding"] for doc in processed_docs if "embedding" in doc.metadata])
    labels = [doc.metadata.get("cluster", -1) for doc in processed_docs]
    AgenticChunker.visualize_clusters_with_umap(embeddings, labels, processed_docs)


