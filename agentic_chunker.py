from typing import List, Union
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
import uuid
from dotenv import load_dotenv
import os
from sklearn.pipeline import Pipeline
import hdbscan
from sentence_transformers import SentenceTransformer
from umap import UMAP
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

class AgenticChunker:
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
        # Need to set up the OpenAI API client here if you want to use it for generating summaries.

        self.model = SentenceTransformer(model_name)

    # Need a function that loops through each chunk and adds the following metadata:
    # - chunk_id: A unique identifier for the chunk, e.g., "source_0", "source_1", etc.
    # - chunk/proposition: The chunk of text itself.
    # - summary: A summary heading for the chunk, to be derived in another function via an API call to OpenAI.
    # - title: A custom title for the chunk, to be derived in another function via an API call.
    # - returns a list of Document objects with the metadata added.

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of Document objects into smaller pieces.
        
        Args:
            documents: A list of Document objects to be chunked.

        Returns:
            A list of Document objects with chunked content and metadata.
        """
        chunked_docs = []
        for doc in documents:
            chunks = self.splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())[:self.id_length_lim]  # Generate a short unique ID for the chunk
                summary = self.generate_summary(chunk)
                chunked_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": chunk_id,
                            "chunk/proposition": chunk,
                            "summary": summary,
                            **doc.metadata
                        }
                    )
                )
        return chunked_docs
    
    # Need a function that generates a summary of the chunk passed into it, using the OpenAI API.
    def generate_summary(self, chunk: str) -> str:
        """
        Generate a summary for a given chunk of text.
        
        Args:
            chunk: The text chunk to summarize.

        Returns:
            A summary of the chunk.
        """

        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
             """
            You are a helpful assistant that summarizes chunks of text from a specific document.
            Your task is to read the content of each chunk and generate a concise summary that captures the main points.
            The summary should be informative and relevant to the content of the chunk.

            Example:
            Chunk: "The financial report for Arsenal FC shows a significant increase in revenue, driven by successful merchandise sales and matchday income."
            Summary: "Arsenal FC's financial report highlights a revenue increase due to merchandise and matchday income."

            Another example:
            Chunk: "The new stadium development plans for Arsenal FC have been approved, promising to enhance the matchday experience for fans."
            Summary: "Arsenal FC's new stadium development plans approved to improve matchday experience."

            Only respond with the chunk summary, nothing else.
             """
             ),
            ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}")
        ])

        chain = PROMPT | self.llm

        summary = chain.invoke({
            "proposition": chunk,
            "current_summary": ""
        }).content

        return summary
    
    # Need a function that generates embeddings for each chunk summary, which will be used for clustering.
    def generate_embeddings(self, chunks: List[Document]) -> np.ndarray:
        """
        Generate embeddings for the summaries of the provided chunks using a pre-trained model.
        Args:
            chunks: A list of Document objects, each containing a summary in its metadata.
        Returns:
            A numpy array of embeddings for the summaries.
        """
        if not chunks:
            return np.empty((0, 0))

        summaries = [chunk.metadata.get("summary", "") for chunk in chunks]

        # Generate embeddings for the summaries
        embeddings = self.model.encode(summaries, convert_to_tensor=False)

        return np.array(embeddings)
    
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

        if embeddings.shape[0] == 0:
            return []

        clustering_model = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        labels = clustering_model.fit_predict(embeddings)

        clusters = {}

        for label, chunk in zip(labels, chunks):
            # Optionally skip noise points (label = -1)
            if label == -1:
                continue
            clusters.setdefault(label, []).append(chunk)

        return list(clusters.values())
    
    # Need a function that projects the clustered chunks into a visualization space using UMAP
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

        # Optional: annotate noise
        if -1 in labels:
            noise_indices = [i for i, l in enumerate(labels) if l == -1]
            plt.scatter(
                embedding_2d[noise_indices, 0],
                embedding_2d[noise_indices, 1],
                c='lightgray', label='Noise', s=60, edgecolors='k'
            )

        # Optional legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}',
                            markerfacecolor=palette(unique_labels.index(label)), markersize=8)
                for label in unique_labels if label != -1]
        if -1 in unique_labels:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Noise',
                                    markerfacecolor='lightgray', markersize=8))
        plt.legend(handles=handles, loc='best')

        plt.tight_layout()
        plt.show()

