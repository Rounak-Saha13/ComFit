import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from schemas import Conversation, ConversationCreate, ConversationResponse
from database import supabase
from auth import optional_verify_token

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/conversations", response_model=List[Conversation])
async def list_convos(user_id: Optional[str] = Depends(optional_verify_token)):
    try:
        if not user_id:
            logger.info("list_convos: guest user, returning empty list")
            return []
        logger.debug("list_convos: fetching conversations for user_id=%s", user_id)
        resp = (supabase.table("conversations") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .execute()
        )
        logger.debug("list_convos: query response: %s", resp)
        return resp.data or []
    except Exception as e:
        logger.exception("list_convos: database error for user_id=%s", user_id)
        raise HTTPException(500, f"Database error: {str(e)}")

@router.post("/conversations", response_model=ConversationResponse)
async def create_convo (
    convo: ConversationCreate,
    user_id: str = Depends(optional_verify_token),
):
    try:
        conversation_id = str(uuid.uuid4())
        # persist if real user
        if user_id:
            logger.debug("create_convo: creating conversation for user_id=%s", user_id)
            logger.debug("create_convo: payload=%s", convo)
            row = {
                "id": conversation_id,
                "title": convo.title,
                "user_id": user_id,
            }
            resp = supabase.table("conversations").insert(row).execute()
            if not resp.data:
                logger.error("create_convo: insert returned no data, resp=%s", resp)
                raise HTTPException(500, "Failed to create conversation")
            logger.info("create_convo: created conversation %s for user %s", conversation_id, user_id)
            return resp.data[0]
        
        # for guest user
        logger.info("create_convo: guest user, returning ephemeral convo %s", conversation_id)
        return ConversationResponse(
            id=conversation_id,
            title=convo.title,
            user_id=None
        )
    except Exception as e:
        logger.exception("create_convos: error in creating conversation")
        raise HTTPException(500, f"Database error: {str(e)}")

@router.patch("/conversations/{cid}", response_model=Conversation)
async def rename_convo(
    cid: str,
    convo: ConversationCreate,
    user_id: Optional[str] = Depends(optional_verify_token)
):
    if not user_id:
        logger.warning("rename_convo: guest user renaming %s to %s", cid, convo.title)
        return Conversation(id=cid, title=convo.title, user_id=None)
    try:
        logger.debug("rename_convo: updating convo %s title to %s for user %s", cid, convo.title, user_id)
        resp = (
            supabase.table("conversations") \
            .update({"title": convo.title}) \
            .eq("id", cid) \
            .eq("user_id", user_id) \
            .execute()
        )
        if not resp.data:
            logger.warning("rename_convo: convo not found or not owned by user %s", user_id)
            raise HTTPException(404, "Conversation not found")
        logger.info("rename_convo: renamed convo %s", cid)        
        return resp.data[0]
    except Exception as e:
        logger.exception("rename_convo: error renaming convo %s for user %s", cid, user_id)
        raise HTTPException(500, f"Database error: {str(e)}")

@router.delete("/conversations/{cid}")
async def delete_convo(cid: str, user_id: Optional[str] = Depends(optional_verify_token)):
    try:
        logger.debug("delete_convo: deleting convo %s for user %s", cid, user_id)
        resp = (
            supabase.table("conversations") \
            .delete() \
            .eq("id", cid) \
            .eq("user_id", user_id) \
            .execute()
        )
        logger.info("delete_convo: deleted convo %s", cid)
        return {"detail": "deleted"}
    except Exception as e:
        logger.exception("delete_convo: error deleting convo %s for user %s", cid, user_id)
        raise HTTPException(500, f"Database error: {str(e)}")
    
# test for debugging
@router.get("/test-db")
async def test_db():
    try:
        resp = supabase.table("conversations").select("count").execute()
        return {"status": "Database connection working", "response": str(resp)}
    except Exception as e:
        return {"status": "Database connection failed", "error": str(e)}
