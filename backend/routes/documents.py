import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from auth import optional_verify_token
from database import supabase
from chat_engine.document_manager import document_manager
from schemas import DocumentAppendResponse

router = APIRouter()

@router.post("/upload", response_model=DocumentAppendResponse)
async def upload_document(
    file: UploadFile = File(...),
    preset: str = Form("default"),
    user_id: Optional[str] = Depends(optional_verify_token)
):
    """
    Upload and process a PDF document for RAG indexing.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        
        # Save uploaded file temporarily
        temp_path = f"temp_{filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Process PDF
            processed_path = document_manager.process_pdf(temp_path)
            if not processed_path:
                raise HTTPException(status_code=500, detail="Failed to process PDF")
            
            # Create vector index
            success = document_manager.create_index(processed_path, preset)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create vector index")
            
            # Save document metadata to database if user is authenticated
            if user_id and supabase:
                try:
                    doc_metadata = {
                        "id": file_id,
                        "user_id": user_id,
                        "filename": file.filename,
                        "preset": preset,
                        "processed_path": processed_path,
                        "file_size": len(content)
                    }
                    supabase.table("documents").insert(doc_metadata).execute()
                except Exception as e:
                    print(f"Warning: Failed to save document metadata: {e}")
            
            return {"detail": f"Document '{file.filename}' uploaded and indexed successfully for preset '{preset}'"}
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/list")
async def list_documents(
    user_id: Optional[str] = Depends(optional_verify_token)
):
    """
    List all available documents and indexes.
    """
    try:
        documents = document_manager.list_documents()
        indexes = document_manager.list_indexes()
        
        # Filter documents by user if authenticated
        if user_id and supabase:
            try:
                user_docs = supabase.table("documents").select("*").eq("user_id", user_id).execute()
                user_doc_ids = {doc["id"] for doc in user_docs.data}
                documents = [doc for doc in documents if any(user_doc_id in doc["name"] for user_doc_id in user_doc_ids)]
            except Exception as e:
                print(f"Warning: Failed to filter documents by user: {e}")
        
        return {
            "documents": documents,
            "indexes": indexes
        }
        
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.delete("/{filename}")
async def delete_document(
    filename: str,
    user_id: Optional[str] = Depends(optional_verify_token)
):
    """
    Delete a document and its associated index.
    """
    try:
        # Check if user owns the document
        if user_id and supabase:
            try:
                user_docs = supabase.table("documents").select("*").eq("user_id", user_id).execute()
                user_doc_names = {doc["filename"] for doc in user_docs.data}
                if filename not in user_doc_names:
                    raise HTTPException(status_code=403, detail="Not authorized to delete this document")
            except Exception as e:
                print(f"Warning: Failed to verify document ownership: {e}")
        
        success = document_manager.delete_document(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from database if authenticated
        if user_id and supabase:
            try:
                supabase.table("documents").delete().eq("filename", filename).eq("user_id", user_id).execute()
            except Exception as e:
                print(f"Warning: Failed to remove document from database: {e}")
        
        return {"detail": f"Document '{filename}' deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.post("/query")
async def query_documents(
    query: str = Form(...),
    preset: str = Form("default"),
    user_id: Optional[str] = Depends(optional_verify_token)
):
    """
    Query documents using the vector index.
    """
    try:
        # Check if user has access to the preset
        if user_id and supabase:
            try:
                user_docs = supabase.table("documents").select("*").eq("user_id", user_id).eq("preset", preset).execute()
                if not user_docs.data:
                    raise HTTPException(status_code=403, detail="Not authorized to access this preset")
            except Exception as e:
                print(f"Warning: Failed to verify preset access: {e}")
        
        result = document_manager.query_index(query, preset)
        if result is None:
            raise HTTPException(status_code=404, detail="No documents found for this preset")
        
        return {"result": result}
        
    except Exception as e:
        print(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/presets")
async def list_presets(
    user_id: Optional[str] = Depends(optional_verify_token)
):
    """
    List all available presets (indexes).
    """
    try:
        indexes = document_manager.list_indexes()
        
        # Filter presets by user if authenticated
        if user_id and supabase:
            try:
                user_presets = supabase.table("documents").select("preset").eq("user_id", user_id).execute()
                user_preset_set = {doc["preset"] for doc in user_presets.data}
                indexes = [preset for preset in indexes if preset in user_preset_set]
            except Exception as e:
                print(f"Warning: Failed to filter presets by user: {e}")
        
        return {"presets": indexes}
        
    except Exception as e:
        print(f"Error listing presets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list presets: {str(e)}")