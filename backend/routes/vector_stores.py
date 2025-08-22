import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

@router.get("/vector-stores", tags=["vector-stores"])
async def list_vector_stores():
    """List all available vector stores (.duckdb files) in the configured directory"""
    try:
        # Get vector stores path from environment variable
        vector_stores_path = os.environ.get("VECTOR_STORES_PATH", "/app/vector_stores")
        
        # Convert to Path object
        stores_dir = Path(vector_stores_path)
        
        if not stores_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Vector stores directory not found: {vector_stores_path}"
            )
        
        # Find all .duckdb files
        duckdb_files = list(stores_dir.glob("*.duckdb"))
        
        if not duckdb_files:
            return {"vector_stores": [], "message": "No vector stores found"}
        
        # Create friendly names for each vector store
        vector_stores = []
        for file_path in duckdb_files:
            # Remove .duckdb extension and convert to friendly name
            filename = file_path.stem
            
            # Create a more readable display name
            display_name = filename.replace("_", " ").replace("-", " ")
            
            # Handle acronyms and capitalization
            words = display_name.split()
            formatted_words = []
            for word in words:
                # If word is all caps (like CFIR), keep it as is
                if word.isupper():
                    formatted_words.append(word)
                # Otherwise, capitalize first letter only
                else:
                    formatted_words.append(word.capitalize())
            
            display_name = " ".join(formatted_words)
            
            vector_stores.append({
                "id": filename,
                "display_name": display_name,
                "filename": file_path.name,
                "path": str(file_path)
            })
        
        return {
            "vector_stores": vector_stores,
            "directory": str(stores_dir),
            "count": len(vector_stores)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list vector stores: {str(e)}"
        )