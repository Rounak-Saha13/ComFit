from fastapi import APIRouter, Depends, HTTPException
from schemas import BranchBase
from database import supabase
from auth import optional_verify_token
from typing import List, Dict, Any

router = APIRouter(prefix="/branches")

@router.post("/{branch_id}/activate", response_model = dict)
async def activate_branch(branch_id: str, user_id: str = Depends(optional_verify_token)):
    branch_resp = supabase.table("branches") \
        .select("conversation_id, edit_at_id") \
        .eq("id", branch_id) \
        .execute()
    
    if not branch_resp.data:
        raise HTTPException(404, "Branch not found")
    branch = branch_resp.data[0]
    conversation_id = branch["conversation_id"]
    edit_at_id = branch["edit_at_id"]

    supabase.table("branches") \
        .update({"is_active": False}) \
        .eq("conversation_id", conversation_id) \
        .eq("edit_at_id", edit_at_id) \
        .execute()
    
    update_resp = supabase.table("branches") \
        .update({"is_active": True}) \
        .eq("id", branch_id) \
        .execute()
    
    if not update_resp.data:
        raise HTTPException(500, "Failed to activate branch")
    return {"detail": "Branch activated"}

@router.patch("/{branch_id}", response_model = dict)
async def update_branch(
    branch_id: str,
    messages: List[Dict[str, Any]],
    user_id: str = Depends(optional_verify_token),
):
    """Update branch messages"""
    if not user_id:
        raise HTTPException(401, "Authentication required")
    
    # Verify the branch exists and user has access
    branch_resp = supabase.table("branches") \
        .select("*, conversations!inner(user_id)") \
        .eq("id", branch_id) \
        .eq("conversations.user_id", user_id) \
        .execute()
    
    if not branch_resp.data:
        raise HTTPException(404, "Branch not found or access denied")

    # Update the branch messages
    update_resp = supabase.table("branches") \
        .update({"messages": messages}) \
        .eq("id", branch_id) \
        .execute()
    
    if not update_resp.data:
        raise HTTPException(500, "Failed to update branch")
    
    return {"detail": "Branch updated"}
