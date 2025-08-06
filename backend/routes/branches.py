from fastapi import APIRouter, Depends, HTTPException
from schemas import BranchBase
from database import supabase
from auth import optional_verify_token

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