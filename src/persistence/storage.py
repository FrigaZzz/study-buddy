from typing import Dict, Any, Optional, List
import json
import os
import aiofiles
from abc import ABC, abstractmethod

class Storage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def save(self, collection: str, key: str, data: Dict[str, Any]) -> bool:
        """Save data to storage."""
        pass
    
    @abstractmethod
    async def load(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """Load data from storage."""
        pass
    
    @abstractmethod
    async def delete(self, collection: str, key: str) -> bool:
        """Delete data from storage."""
        pass
    
    @abstractmethod
    async def list_keys(self, collection: str) -> List[str]:
        """List all keys in a collection."""
        pass


class FileStorage(Storage):
    """File-based storage implementation."""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize file storage.
        
        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    async def save(self, collection: str, key: str, data: Dict[str, Any]) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            collection: Collection name (used as directory)
            key: Key for the data
            data: Data to save
            
        Returns:
            Success status
        """
        collection_dir = os.path.join(self.base_dir, collection)
        os.makedirs(collection_dir, exist_ok=True)
        
        file_path = os.path.join(collection_dir, f"{key}.json")
        
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    async def load(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            collection: Collection name (used as directory)
            key: Key for the data
            
        Returns:
            Loaded data or None if not found
        """
        file_path = os.path.join(self.base_dir, collection, f"{key}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            return json.loads(content)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    async def delete(self, collection: str, key: str) -> bool:
        """
        Delete a JSON file.
        
        Args:
            collection: Collection name (used as directory)
            key: Key for the data
            
        Returns:
            Success status
        """
        file_path = os.path.join(self.base_dir, collection, f"{key}.json")
        
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error deleting data: {e}")
            return False
    
    async def list_keys(self, collection: str) -> List[str]:
        """
        List all keys in a collection.
        
        Args:
            collection: Collection name (used as directory)
            
        Returns:
            List of keys
        """
        collection_dir = os.path.join(self.base_dir, collection)
        
        if not os.path.exists(collection_dir):
            return []
        
        try:
            files = os.listdir(collection_dir)
            return [f[:-5] for f in files if f.endswith('.json')]
        except Exception as e:
            print(f"Error listing keys: {e}")
            return []