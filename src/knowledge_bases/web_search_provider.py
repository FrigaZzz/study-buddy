from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
import aiohttp

class WebSearchProvider:
    """Provides web search capabilities."""
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """
        Initialize the web search provider.
        
        Args:
            api_key: API key for search service
            config: Configuration parameters
        """
        self.api_key = api_key
        self.config = config or {}
        self.search_url = self.config.get("search_url", "https://api.search.com/v1/search")
        
    async def search(self, topic: str, subtopic: Optional[str] = None) -> List[Document]:
        """
        Search the web for relevant content.
        
        Args:
            topic: Main topic
            subtopic: Optional subtopic for more specific search
            
        Returns:
            List of documents from search results
        """
        query = f"{topic}"
        if subtopic:
            query += f" {subtopic}"
            
        # Get search parameters from config
        num_results = self.config.get("num_results", 5)
        
        # Perform search
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.search_url,
                params={
                    "q": query,
                    "num": num_results,
                    "key": self.api_key
                }
            ) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
        # Convert search results to documents
        documents = []
        for result in data.get("results", []):
            doc = Document(
                page_content=result.get("snippet", ""),
                metadata={
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "source": "web_search"
                }
            )
            documents.append(doc)
            
        return documents