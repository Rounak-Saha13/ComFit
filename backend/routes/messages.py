import logging
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Body, Request

from schemas import Message, MessageCreate, MessageUpdate, History, FeedbackRequest
from database import supabase
from auth import verify_token, optional_verify_token
from chat_engine import chat_answer
from typing import List, Dict, Optional, Any
from .chat import check_guest_rate_limit
from pydantic import BaseModel

router = APIRouter(prefix="/messages")
logger = logging.getLogger(__name__)

class RegenerateRequest(BaseModel):
    history: List[Dict[str, Any]]
    model: str
    preset: str
    temperature: float
    rag_method: Optional[str] = None
    retrieval_method: Optional[str] = None

@router.post("/", response_model=Message)
async def create_message(msg: MessageCreate, user_id: str = Depends(verify_token)):
    logger.debug("create_message: inserting message %s for user %s", msg, user_id)
    insert_resp = supabase.table("messages") \
        .insert(msg.model_dump()) \
        .execute()
    if not insert_resp.data:
        logger.error("create_message: failed to insert message, resp=%s", insert_resp)
        raise HTTPException(500, "Branch Error: Could not insert message")
    created = insert_resp.data[0]
    logger.info("create_message: inserted message id=%s", created.get("id"))
    
    # Convert database field names to Pydantic model field names
    if "thinking_time" in created:
        created["thinkingTime"] = created.pop("thinking_time")
    
    return created

@router.get("/conversation/{cid}", response_model=List[Message])
async def get_messages(cid: str, user_id: str = Depends(verify_token)):
    try:
        conv_resp = supabase.table("conversations") \
            .select("id") \
            .eq("id", cid) \
            .eq("user_id", user_id) \
            .execute()

        if not conv_resp.data:
            raise HTTPException(404, "Conversation not found or access denied")

        resp = supabase.table("messages") \
            .select("*") \
            .eq("conversation_id", cid) \
            .order("created_at", desc=False) \
            .execute()

        logger.debug("get_messages: fetched %d messages for conversation %s", 
                     len(resp.data or []), cid)
        
        # Convert database field names to Pydantic model field names
        messages = resp.data or []
        for msg in messages:
            if "thinking_time" in msg:
                msg["thinkingTime"] = msg.pop("thinking_time")
        
        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_messages: error fetching messages: %s", e)
        raise HTTPException(500, f"Failed to fetch messages: {str(e)}")

@router.patch("/{mid}", response_model=Message)
async def edit_message(mid: str, body: MessageUpdate, user_id: str = Depends(verify_token)):
    try:
        resp = supabase.table("messages") \
            .select("*, conversations!inner(user_id)") \
            .eq("id", mid) \
            .eq("conversations.user_id", user_id) \
            .execute()

        if not resp.data:
            raise HTTPException(404, "Message not found or access denied")

        # Only update fields that are not None
        update_data = {k: v for k, v in body.model_dump().items() if v is not None}
        
        update_resp = supabase.table("messages") \
            .update(update_data) \
            .eq("id", mid) \
            .execute()

        if not update_resp.data:
            raise HTTPException(500, "Failed to update message")

        # Convert database field names to Pydantic model field names
        message_data = update_resp.data[0]
        if "thinking_time" in message_data:
            message_data["thinkingTime"] = message_data.pop("thinking_time")
        
        return message_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error("edit_message: failed to edit message %s, error=%s", mid, e)
        raise HTTPException(500, f"Failed to update message: {str(e)}")

@router.post("/{mid}/feedback", response_model=dict)
async def feedback(
    mid: str,
    request: Request,
    payload: FeedbackRequest = Body(...),
    user_id: Optional[str] = Depends(optional_verify_token),
):
    check_guest_rate_limit(request, user_id)
    if user_id:
        supabase.table("messages") \
            .update({"feedback": payload.rating}) \
            .eq("id", mid) \
            .execute()
    return {"detail": "feedback added"}

@router.post("/{mid}/regenerate", response_model=Message)
async def regenerate(
    mid: str,
    request: Request,
    payload: Optional[RegenerateRequest] = Body(None),
    user_id: Optional[str] = Depends(optional_verify_token),
):
    # Temporarily disable rate limiting for testing
    # check_guest_rate_limit(request, user_id)

    if user_id:
        # logged-in: fetch original, regenerate, update DB
        resp = supabase.table("messages") \
            .select("*, conversations!inner(user_id)") \
            .eq("id", mid) \
            .eq("conversations.user_id", user_id) \
            .execute()
        if not resp.data:
            raise HTTPException(404, "Message not found or access denied")

        ai_msg = resp.data[0]
        convo_id = ai_msg["conversation_id"]
        model = ai_msg["model"]
        preset = ai_msg["preset"]
        temperature = ai_msg["temperature"]
        rag_method = ai_msg["method"]
        retrieval_method = ai_msg["retrieval_method"]

        history_resp = supabase.table("messages") \
            .select("id, content, sender") \
            .eq("conversation_id", convo_id) \
            .order("created_at", desc=False) \
            .execute()
        if not history_resp.data:
            raise HTTPException(404, "Conversation history not found")

        formatted = [
            {"role": "user" if m["sender"] == "user" else "assistant",
             "content": m["content"]}
            for m in history_resp.data
        ]

        new_content, duration = await chat_answer(
            formatted, convo_id, model, preset, temperature, user_id,
            payload.rag_method or "No Specific RAG Method",
            payload.retrieval_method or "local context only"
        )

        update_resp = supabase.table("messages") \
            .update({"content": new_content, "thinking_time": duration}) \
            .eq("id", mid) \
            .execute()
        if not update_resp.data:
            raise HTTPException(500, "Failed to update message")

        # Convert database field names to Pydantic model field names
        message_data = update_resp.data[0]
        if "thinking_time" in message_data:
            message_data["thinkingTime"] = message_data.pop("thinking_time")
        
        return message_data

    else:
        # guest: require body.history, regenerate in memory
        if not payload or not payload.history:
            raise HTTPException(400, "Guests must include `history` in the request body")
        new_content, duration = await chat_answer(
            payload.history, mid, payload.model, payload.preset, payload.temperature, user_id,
            payload.rag_method or "No Specific RAG Method",
            payload.retrieval_method or "local context only"
        )
        return Message(
            id=mid,
            content=new_content,
            thinkingTime=duration,
            conversation_id=None,
            sender="assistant",
        )

@router.post("/conversations/{cid}/branches", response_model=dict)
async def create_branch(
    cid: str,
    request: Request,
    edit_at_id: str = Body(...),
    messages: List[Dict[str, Any]] = Body(...),
    user_id: Optional[str] = Depends(optional_verify_token),
):
    check_guest_rate_limit(request, user_id)
    if not user_id:
        return {"detail": "Branches only available for logged in users"}

    conv_resp = supabase.table("conversations") \
        .select("id") \
        .eq("id", cid) \
        .eq("user_id", user_id) \
        .execute()
    if not conv_resp.data:
        raise HTTPException(404, "Branch Error: Conversation not found")

    # deactivate existing active branch for this edit
    supabase.table("branches") \
        .update({"is_active": False}) \
        .eq("conversation_id", cid) \
        .eq("edit_at_id", edit_at_id) \
        .execute()

    # Format messages to ensure they have all required fields
    formatted_messages = []
    for msg in messages:
        formatted_msg = {
            "id": msg.get("id"),
            "conversation_id": cid,
            "sender": msg.get("sender"),
            "content": msg.get("content"),
            "thinking_time": msg.get("thinkingTime", 0),
            "created_at": msg.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "feedback": msg.get("feedback"),
            "model": msg.get("model"),
            "preset": msg.get("preset"),
            "temperature": msg.get("temperature"),
            "top_p": msg.get("top_p"),
            "rag_method": msg.get("rag_method"),
            "retrieval_method": msg.get("retrieval_method")
        }
        formatted_messages.append(formatted_msg)

    # insert new active branch
    branch_resp = supabase.table("branches").insert({
        "conversation_id": cid,
        "edit_at_id": edit_at_id,
        "messages": formatted_messages,
        "is_original": False,
        "is_active": True,
    }).execute()
    if not branch_resp.data:
        raise HTTPException(500, "Failed to create branch")
    
    # Get the ID from the inserted data
    branch_id = branch_resp.data[0]["id"]
    return {"branch_id": branch_id}

@router.patch("/branches/{branch_id}", response_model=dict)
async def update_branch(
    branch_id: str,
    messages: List[Dict[str, Any]] = Body(...),
    user_id: str = Depends(verify_token),
):
    branch_resp = supabase.table("branches") \
        .select("*, conversations!inner(user_id)") \
        .eq("id", branch_id) \
        .eq("conversations.user_id", user_id) \
        .execute()
    if not branch_resp.data:
        raise HTTPException(404, "Branch not found")

    update_resp = supabase.table("branches") \
        .update({"messages": messages}) \
        .eq("id", branch_id) \
        .execute()
    if not update_resp.data:
        raise HTTPException(500, "Failed to update branch")
    return {"detail": "Branch updated"}

@router.get("/conversation/{cid}/history", response_model=History)
async def get_history(cid: str, user_id: Optional[str] = Depends(optional_verify_token)):
    if not user_id:
        return {
            "messages": [],
            "branchesByEditId": {},
            "currentBranchIndexByEditId": {},
        }

    current_branch_resp = supabase.table("branches") \
        .select("id, messages, edit_at_id") \
        .eq("conversation_id", cid) \
        .eq("is_active", True) \
        .execute()
    if current_branch_resp.data:
        messages = current_branch_resp.data[0]["messages"]
    else:
        msg_resp = supabase.table("messages") \
            .select("id, content, sender, thinking_time, created_at") \
            .eq("conversation_id", cid) \
            .order("created_at", desc=False) \
            .execute()
        messages = msg_resp.data or []

    # ensure conversation_id on every message and convert field names
    for m in messages:
        m["conversation_id"] = cid
        if "thinking_time" in m:
            m["thinkingTime"] = m.pop("thinking_time")

    # fetch all branches for sidebar
    branch_resp = supabase.table("branches") \
        .select("id, edit_at_id, messages, is_original") \
        .eq("conversation_id", cid) \
        .order("created_at", desc=False) \
        .execute()
    branch_objs = branch_resp.data or []

    branches_by_edit: Dict[str, List[List[Message]]] = {}
    for b in branch_objs:
        edit_id = b["edit_at_id"]
        for msg in b["messages"]:
            msg["conversation_id"] = cid
            if "thinking_time" in msg:
                msg["thinkingTime"] = msg.pop("thinking_time")
        branches_by_edit.setdefault(edit_id, []).append({
            "messages": b["messages"],
            "branchId": b["id"],
            "isOriginal": b["is_original"]
        })

    current_index_map = { eid: len(lst) - 1 for eid, lst in branches_by_edit.items() }

    # Get original messages from database (not from branches)
    original_msg_resp = supabase.table("messages") \
        .select("id, content, sender, thinking_time, created_at") \
        .eq("conversation_id", cid) \
        .order("created_at", desc=False) \
        .execute()
    original_messages = original_msg_resp.data or []
    
    # Convert field names for original messages
    for msg in original_messages:
        msg["conversation_id"] = cid
        if "thinking_time" in msg:
            msg["thinkingTime"] = msg.pop("thinking_time")

    return {
        "messages": messages,
        "originalMessages": original_messages,
        "branchesByEditId": branches_by_edit,
        "currentBranchIndexByEditId": current_index_map,
    }