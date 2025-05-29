from typing import List, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI
import uuid
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class AgenticChunker:
    def __init__(self, openai_api_key: Union[str, None] = None):
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

    # Need a function that creates the title of the chunk passed into it, using the OpenAI API.
    def generate_title(self, chunk: str) -> str:
        """
        Generate a title for a given chunk of text.
        This is a placeholder function that should be replaced with an actual API call to OpenAI or similar service.
        
        Args:
            chunk: The text chunk to title.

        Returns:
            A title for the chunk.
        """

        PROMPT = ChatPromptTemplate.from_messages([
            ("system",
             """
            You are a helpful assistant that oversees chunks of text from a specific document.
            Your task is to assess the content of each chunk and generate a concise title that captures the essence of the text.
            The title should be informative, engaging, and relevant to the content of the chunk.
            If you feel that various chunks are related, you can create a title that encompasses the main theme or topic of those chunks.

            Example:
            Chunk: "The financial report for Arsenal FC shows a significant increase in revenue, driven by successful merchandise sales and matchday income."
            Title: "Arsenal FC's Financial Growth: Revenue Surge from Merchandise and Matchday Income"

            A similar example chunk might be:
            Chunk: "Arsenal FC's financial performance has been impressive this year, with a notable rise in income from merchandise and matchday activities."
            Title: "Arsenal FC's Financial Performance: Rise in Merchandise and Matchday Income"
            Here the chunks are similar, so a common title has been created for both chunks.

            A different example chunk might be:
            Chunk: "The new stadium development plans for Arsenal FC have been approved, promising to enhance the matchday experience for fans."
            Title: "Arsenal FC's New Stadium Development Plans Approved"
            Here the chunk is not directly related to the previous example chunks, so a new title has been created for this chunk.

            Only respond with the chunk title, nothing else.
             """
             ),
            ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}")
        ])

        chain = PROMPT | self.llm

        response = chain.invoke({
            "proposition": chunk,
            "current_summary": self.generate_summary(chunk),
            "current_title": "Untitled"
        })
        # Placeholder for actual API call
        return f"Title for chunk: {chunk[:30]}..."
    
    # Need a function that generates a summary of the chunk passed into it, using the OpenAI API.
    def generate_summary(self, chunk: str) -> str:
        """
        Generate a summary for a given chunk of text.
        This is a placeholder function that should be replaced with an actual API call to OpenAI or similar service.
        
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
        })

        return summary
    
    # Need a function that creates a clustering algorithm that uses the embeddings of the chunk summaries to group similar chunks together.
    def cluster_chunks(self, chunks: List[Document]) -> List[List[Document]]:
        """
        Cluster chunks based on their summaries using embeddings.
        
        Args:
            chunks: A list of Document objects with summaries.

        Returns:
            A list of clusters, where each cluster is a list of Document objects.
        """
        
        # This could involve generating embeddings for the summaries and then applying a clustering algorithm like KMeans or DBSCAN.
        return [chunks]  # For now, return all chunks in a single cluster