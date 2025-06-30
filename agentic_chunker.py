from typing import List, Union
import numpy as np
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
import asyncio
import uuid
from dotenv import load_dotenv
import os
import hdbscan
from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

class AgenticChunker:

    PROMPT = ChatPromptTemplate.from_messages([

        ("system",
            """
        You are a helpful assistant that summarizes chunks of text from a specific document.
        Your task is to read the content of each chunk and generate a concise summary that captures the main points.
        The summary should be informative and relevant to the content of the chunk.

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

        self.llm = ChatOpenAI(model='gpt-4-1106-preview', openai_api_key=openai_api_key, temperature=0)

        self.model = SentenceTransformer(model_name)

    # Need a function that applies the other functions to ultimately generate a final set of embeddings for the chunk summaries
    async def get_embeddings(self, documents: List[Document], chunk_overlap: int = 0) -> List[Document]:
        """
        Process a list of Document objects by chunking, summarizing, and clustering them.
        
        Args:
            documents: A list of Document objects to be processed.
            chunk_overlap: The number of overlapping tokens between chunks.

        Returns:
            A NumPy array of embeddings for the clustered summaries.
        """
        chunked_docs = await self.chunk_documents(documents, chunk_overlap)
        clusters = self.cluster_chunks(chunked_docs)
        return self.generate_embeddings([doc for cluster in clusters for doc in cluster])

    # Need a function that loops through each chunk and adds the following metadata:
    # - chunk_id: A unique identifier for the chunk, e.g., "source_0", "source_1", etc.
    # - chunk/proposition: The chunk of text itself.
    # - summary: A summary heading for the chunk, to be derived in another function via an API call to OpenAI.
    # - title: A custom title for the chunk, to be derived in another function via an API call.
    # - returns a list of Document objects with the metadata added.

    async def chunk_documents(self, documents: List[Document], chunk_overlap: int = 0) -> List[Document]:
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
            model_name="all-MiniLM-L6-v2"
        )
        tokens = text_splitter.split_documents(text_docs)

        chunked_docs = []

        # Process non-table chunks with LLM summarization
        summaries = await asyncio.gather(*[
            self.generate_summary(token.page_content) for token in tokens
        ])

        for i, (token, summary) in enumerate(zip(tokens, summaries)):
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
    
    # Need a function that generates a summary of the chunk passed into it, using the OpenAI API.
    async def generate_summary(self, chunk: str) -> str:
        """
        Generate a summary for a given chunk of text.
        
        Args:
            chunk: The text chunk to summarize.

        Returns:
            A summary of the chunk.
        """

        chain = self.PROMPT | self.llm

        return (await chain.ainvoke({"chunk": chunk})).content
    
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
        embeddings = self.generate_embeddings(chunks)

        if not chunks:
            return []

        clustering_model = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        labels = clustering_model.fit_predict(embeddings)

        clusters = {}

        for label, chunk in zip(labels, chunks):
            chunk.metadata["cluster"] = int(label)
            if label == -1:
                continue
            clusters.setdefault(label, []).append(chunk)

        return list(clusters.values())
    
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


