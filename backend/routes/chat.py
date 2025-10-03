from fastapi import APIRouter, HTTPException, Depends, Header, Body
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import time
import re
from fastapi.concurrency import run_in_threadpool
from dateutil import parser 
import logging

# --- CRITICAL FIX: Changing to absolute import based on package structure ---
# The ChatEngine class is located in chat_engine/client.py.
# This assumes the 'chat_engine' package is accessible from the root context (main.py).
from chat_engine.client import ChatEngine 
from schemas import Message, MessageUpdate, FeedbackRequest, TitleRequest, History, BranchItem, ChatRequest
from database import supabase, supabase_auth

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Global Chat Engine Initialization ---
try:
    # Initialize the ChatEngine instance once on startup
    chat_engine = ChatEngine() 
    logger.info("ChatEngine initialized successfully within router.")
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize ChatEngine: {e}", exc_info=True)
    chat_engine = None # Set to None if initialization fails

# --- Utility Functions (Unchanged) ---
def normalize_sender(s: str) -> str:
    s = (s or "").lower()
    if s in ("assistant", "bot", "ai"):
        return "ai"
    return "user" if s == "user" else s

def role_for_llm(sender: str) -> str:
    return "assistant" if normalize_sender(s) == "ai" else "user"

def ensure_uuid(s: str) -> str:
    try:
        uuid.UUID(str(s))
        return str(s)
    except Exception:
        return str(uuid.uuid4())

# --- Dependency Injection (Unchanged) ---
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Extract user ID from authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    try:
        # Verify the token and get user info
        user = supabase_auth.auth.get_user(token)
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----------------------------------------------------------------------
# --- NEW/UPDATED ENDPOINTS ---
# ----------------------------------------------------------------------

@router.get("/messages/conversation/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Get conversation history with branches.
    Updated to retrieve the 'image_url' field.
    """
    try:
        logger.debug(f"Loading history for conversation {conversation_id}")
        
        # Verify conversation ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conversation_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get all messages for this conversation (main conversation flow)
        # CRITICAL: Added 'image_url' to the select statement
        messages_result = supabase.table("messages")\
            .select("id, conversation_id, sender, content, thinking_time, feedback, model, preset, system_prompt, speculative_decoding, temperature, top_p, strategy, rag_method, retrieval_method, created_at, image_url")\
            .eq("conversation_id", conversation_id)\
            .order("created_at", desc=False)\
            .execute()
        
        logger.debug(f"Found {len(messages_result.data) if messages_result.data else 0} main messages")
        
        if not messages_result.data:
            return History(
                messages=[],
                originalMessages=[],
                branchesByEditId={},
                currentBranchIndexByEditId={},
                activeBranchId=None
            )
        
        # Get all branches for this conversation
        branches_result = supabase.table("branches")\
            .select("*")\
            .eq("conversation_id", conversation_id)\
            .execute()
        
        logger.debug(f"Found {len(branches_result.data) if branches_result.data else 0} branches")
        
        # Organize messages and branches
        messages = []
        original_messages = []
        branches_by_edit_id = {}
        current_branch_index_by_edit_id = {}
        active_branch_id = None
        
        # Process main conversation messages
        for msg_data in messages_result.data:
            thinking_time = msg_data.get("thinking_time", 0)
            try:
                thinking_time = int(thinking_time)
            except (ValueError, TypeError):
                thinking_time = 0
            
            # CRITICAL: Pass 'image_url' to the Message constructor
            msg = Message(
                id=msg_data["id"],
                conversation_id=msg_data["conversation_id"],
                sender=msg_data["sender"],
                content=msg_data["content"],
                thinking_time=thinking_time,
                feedback=msg_data.get("feedback"),
                model=msg_data.get("model"),
                preset=msg_data.get("preset"),
                system_prompt=msg_data.get("system_prompt"),
                speculative_decoding=msg_data.get("speculative_decoding", False),
                temperature=msg_data.get("temperature", 0.7),
                top_p=msg_data.get("top_p", 1.0),
                strategy=msg_data.get("strategy"),
                rag_method=msg_data.get("rag_method"),
                retrieval_method=msg_data.get("retrieval_method"),
                created_at=parser.isoparse(msg_data["created_at"]),
                image_url=msg_data.get("image_url") # <-- NEW RETRIEVAL
            )
            messages.append(msg)
        
        # Process branches - use JSON messages stored in branches table
        for branch_data in branches_result.data:
            edit_at_id = branch_data["edit_at_id"]
            branch_id = branch_data["id"]
            is_original = branch_data.get("is_original", False)
            is_active = branch_data.get("is_active", False)
            
            logger.debug(f"Processing branch {branch_id} (edit_at: {edit_at_id}, is_original: {is_original}, is_active: {is_active})")
            
            branch_messages_json = branch_data.get("messages", [])
            branch_messages = []
            
            # Convert JSON messages to Message objects
            for msg_json in branch_messages_json:
                thinking_time = msg_json.get("thinking_time", 0)
                try:
                    thinking_time = int(thinking_time)
                except (ValueError, TypeError):
                    thinking_time = 0
                
                created_at_value = msg_json.get("created_at")
                created_at_dt = parser.isoparse(created_at_value) if created_at_value else datetime.utcnow()

                # CRITICAL: Pass 'image_url' from branch JSON to Message constructor
                msg = Message(
                    id=msg_json["id"],
                    conversation_id=msg_json["conversation_id"],
                    sender=msg_json["sender"],
                    content=msg_json["content"],
                    thinking_time=thinking_time,
                    feedback=msg_json.get("feedback"),
                    model=msg_json.get("model"),
                    preset=msg_json.get("preset"),
                    system_prompt=msg_json.get("system_prompt"),
                    speculative_decoding=msg_json.get("speculative_decoding", False),
                    temperature=msg_json.get("temperature", 0.7),
                    top_p=msg_json.get("top_p", 1.0),
                    strategy=msg_json.get("strategy"),
                    rag_method=msg_json.get("rag_method"),
                    retrieval_method=msg_json.get("retrieval_method"),
                    created_at=created_at_dt,
                    image_url=msg_json.get("image_url") # <-- NEW RETRIEVAL
                )
                branch_messages.append(msg)
            
            # Organize by edit point
            if edit_at_id not in branches_by_edit_id:
                branches_by_edit_id[edit_at_id] = []
            
            branch_item = BranchItem(
                branch_id=branch_id if not is_original else None,
                is_original=is_original,
                messages=branch_messages
            )
            
            branches_by_edit_id[edit_at_id].append(branch_item)
            
            # Set current branch index
            if edit_at_id not in current_branch_index_by_edit_id:
                original_index = next((i for i, b in enumerate(branches_by_edit_id[edit_at_id]) if b.is_original), 0)
                current_branch_index_by_edit_id[edit_at_id] = original_index
            
            # Set active branch
            if is_active:
                active_branch_id = branch_id
        
        # Always set original_messages to the main conversation messages
        original_messages = messages.copy()
        
        logger.debug(f"Final structure - messages: {len(messages)}, original: {len(original_messages)}, branches: {len(branches_by_edit_id)}")
        
        # Create History object with proper field mapping
        return History(
            messages=messages,
            originalMessages=original_messages,
            branchesByEditId=branches_by_edit_id,
            currentBranchIndexByEditId=current_branch_index_by_edit_id,
            activeBranchId=active_branch_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

# ----------------------------------------------------------------------

@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """
    Main chat endpoint. Updated to handle and propagate the structured image_url output.
    """
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="AI Service unavailable due to initialization failure.")
        
    try:
        start_time = time.time()
        is_branch_mode = getattr(request, "branch_mode", False)
        
        # --- 1. Save User Message (Updated to use new schema fields) ---
        if not is_branch_mode:
            last_user = next((m for m in reversed(request.messages) if normalize_sender(m.sender) == "user"), None)
            if last_user:
                user_row = {
                    "id": ensure_uuid(last_user.id),
                    "conversation_id": request.conversation_id,
                    "sender": "user",
                    "content": last_user.content,
                    "thinking_time": int(getattr(last_user, "thinking_time", 0) or 0),
                    "feedback": getattr(last_user, "feedback", None),
                    "model": getattr(last_user, "model", request.model),
                    "preset": getattr(last_user, "preset", request.preset),
                    "system_prompt": request.system_prompt,
                    "speculative_decoding": request.speculative_decoding,
                    "temperature": getattr(last_user, "temperature", request.temperature),
                    "top_p": getattr(last_user, "top_p", request.top_p),
                    "strategy": getattr(last_user, "strategy", request.strategy),
                    "rag_method": getattr(last_user, "rag_method", request.rag_method),
                    "retrieval_method": getattr(last_user, "retrieval_method", request.retrieval_method),
                    "user_id": user_id,
                    "image_url": getattr(last_user, "image_url", None), # Persist user image_url if provided
                }
                try:
                    supabase.table("messages").insert(user_row).execute()
                except Exception as e:
                    logger.debug(f"User message insert error (ignoring duplicates): {e}")

        # --- 2. Format Messages for LLM (Unchanged) ---
        formatted_messages = []
        if request.system_prompt:
            formatted_messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            formatted_messages.append({
                "role": role_for_llm(msg.sender),
                "content": msg.content
            })
            
        # --- 3. Call AI Engine and Unpack Structured Response ---
        try:
            # chat_engine.generate_response returns (response_dict, duration_ms)
            response_dict, _duration = await chat_engine.generate_response(
                messages=formatted_messages,
                conversation_id=request.conversation_id,
                model=request.model,
                preset=request.preset,
                temperature=request.temperature,
                user_id=user_id,
                rag_method=request.rag_method,
                retrieval_method=request.retrieval_method,
            )
            
            final_answer = response_dict.get("final_answer", "No answer generated.")
            sources_str = response_dict.get("sources_str", "")
            image_url = response_dict.get("image_url", None) # <-- CRITICAL: CAPTURE IMAGE URL
            
            # Combine text answer and sources for the database `content` field
            ai_content = f"{final_answer.strip()}\n\n{sources_str.strip()}"
            
            if final_answer.startswith("Error:"):
                raise HTTPException(status_code=400, detail=final_answer)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"AI model error during response generation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

        actual_duration = int((time.time() - start_time) * 1000)
        ai_message_id = str(uuid.uuid4())
        created_at_str = datetime.utcnow().isoformat()
        
        # --- 4. Save AI Message to Database (Updated with image_url) ---
        if not is_branch_mode:
            ai_row = {
                "id": ai_message_id,
                "conversation_id": request.conversation_id,
                "sender": "ai",
                "content": ai_content,
                "thinking_time": actual_duration,
                "feedback": None,
                "model": request.model,
                "preset": request.preset,
                "system_prompt": request.system_prompt,
                "speculative_decoding": request.speculative_decoding,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "strategy": request.strategy,
                "rag_method": request.rag_method,
                "retrieval_method": request.retrieval_method,
                "user_id": user_id,
                "image_url": image_url, # <-- CRITICAL: SAVE IMAGE URL TO DB
            }
            inserted = supabase.table("messages").insert(ai_row).execute()
            created_at_str = inserted.data[0]["created_at"] if inserted and inserted.data else created_at_str

        # --- 5. Construct Response Object (Updated with image_url) ---
        ai_message = Message(
            id=ai_message_id,
            conversation_id=request.conversation_id,
            sender="ai",
            content=ai_content,
            thinking_time=actual_duration,
            feedback=None,
            model=request.model,
            preset=request.preset,
            system_prompt=request.system_prompt,
            speculative_decoding=request.speculative_decoding,
            temperature=request.temperature,
            top_p=request.top_p,
            strategy=request.strategy,
            rag_method=request.rag_method,
            retrieval_method=request.retrieval_method,
            created_at=parser.isoparse(created_at_str.replace("Z", "+00:00")) if "Z" in created_at_str else parser.isoparse(created_at_str),
            image_url=image_url, # <-- CRITICAL: Propagate to Message schema
        )

        # NOTE: ChatResponse schema now correctly holds the Message object with image_url
        return ChatResponse(
            result=final_answer, # CLEAN text answer for display
            duration=actual_duration,
            ai_message=ai_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# ----------------------------------------------------------------------
# --- Feedback (Unchanged) ---
# ----------------------------------------------------------------------

@router.post("/messages/{message_id}/feedback")
async def update_message_feedback(
    message_id: str,
    request: FeedbackRequest,
    user_id: str = Depends(get_current_user)
):
    """Update feedback for a message"""
    try:
        # Verify message ownership through conversation
        msg_result = supabase.table("messages")\
            .select("id, conversation_id")\
            .eq("id", message_id)\
            .execute()
        
        if not msg_result.data:
            raise HTTPException(status_code=404, detail="Message not found")
        
        conv_id = msg_result.data[0]["conversation_id"]
        
        # Verify conversation ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conv_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update feedback
        result = supabase.table("messages")\
            .update({"feedback": request.rating})\
            .eq("id", message_id)\
            .execute()
        
        if result.data:
            return {"message": "Feedback updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update feedback")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {str(e)}")

# ----------------------------------------------------------------------
# --- Regenerate (Updated to use structured output) ---
# ----------------------------------------------------------------------

@router.post("/messages/{message_id}/regenerate")
async def regenerate_message(
    message_id: str,
    payload: ChatRequest = Body(...), 
    user_id: Optional[str] = Depends(get_current_user)
):
    """Regenerate an AI message - stateless, returns fresh content only"""
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="AI Service unavailable due to initialization failure.")

    try:
        logger.debug(f"Looking for message with ID: {message_id}")
        
        # First try to find the message in the messages table (including image_url for consistency)
        msg_result = supabase.table("messages")\
            .select("id, conversation_id, content, created_at, sender, image_url")\
            .eq("id", message_id)\
            .in_("sender", ["ai", "assistant"])\
            .execute()
        
        # --- Logic for finding message in main table or branches (Unchanged) ---
        if not msg_result.data:
            request_conversation_id = payload.conversation_id
            branches_result = supabase.table("branches")\
                .select("id, conversation_id, messages, edit_at_id")\
                .eq("conversation_id", request.conversation_id)\
                .execute()
            
            found_message = None
            found_conversation_id = None
            found_created_at = None
            
            for branch in branches_result.data:
                branch_messages = branch.get("messages", [])
                for msg in branch_messages:
                    if msg.get("id") == message_id and msg.get("sender") in ["ai", "assistant"]:
                        found_message = msg
                        found_conversation_id = branch["conversation_id"]
                        found_created_at = msg.get("created_at")
                        break
                if found_message:
                    break
            
            if found_message:
                conv_id = found_conversation_id
                original_created_at = found_created_at
            else:
                raise HTTPException(status_code=404, detail="AI message not found")
        else:
            conv_id = msg_result.data[0]["conversation_id"]
            original_created_at = msg_result.data[0]["created_at"]
        
        # Verify conversation ownership (Unchanged)
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conv_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get conversation history up to this message (Unchanged logic)
        history_result = supabase.table("messages")\
            .select("*")\
            .eq("conversation_id", conv_id)\
            .lt("created_at", original_created_at)\
            .order("created_at", desc=False)\
            .execute()
        
        formatted_messages = []
        last_user_message = None
        for msg in reversed(history_result.data):
            if msg["sender"] == "user":
                last_user_message = msg
                break
        
        if last_user_message:
            formatted_messages.append({
                "role": "user",
                "content": last_user_message["content"]
            })
        
        if payload.system_prompt:
            formatted_messages.insert(0, {
                "role": "system",
                "content": payload.system_prompt
            })
        
        # --- Generate new response using chat engine (CRITICAL CHANGE) ---
        start_time = time.time()
        
        # Call generate_response which returns (response_dict, duration_ms)
        response_dict, _duration = await chat_engine.generate_response(
            messages=formatted_messages,
            conversation_id=conv_id,
            model=payload.model,
            preset=payload.preset,
            temperature=payload.temperature,
            user_id=user_id,
            rag_method=payload.rag_method,
            retrieval_method=payload.retrieval_method
        )
        
        final_answer = response_dict.get("final_answer", "No answer generated.")
        sources_str = response_dict.get("sources_str", "")
        image_url = response_dict.get("image_url", None) # <-- CRITICAL: CAPTURE IMAGE URL
        
        # The full content for the response (answer + sources)
        ai_content = f"{final_answer.strip()}\n\n{sources_str.strip()}"
        
        if final_answer.startswith("Error:"):
            raise HTTPException(status_code=400, detail=final_answer)
                
        actual_duration = int((time.time() - start_time) * 1000)
        
        # Return fresh content without updating database
        response_data = {
            "content": ai_content,
            "duration": actual_duration,
            "model": payload.model,
            "preset": payload.preset,
            "temperature": payload.temperature,
            "rag_method": payload.rag_method,
            "retrieval_method": payload.retrieval_method,
            "thinking_time": actual_duration,
            "image_url": image_url, # <-- NEW FIELD
            "regenerated_content": final_answer, # Clean text answer
        }
        
        logger.debug(f"Returning regeneration response data: {response_data}")
        return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to regenerate message: {str(e)}")

# ----------------------------------------------------------------------
# --- Title Generation (Updated to use correct engine method) ---
# ----------------------------------------------------------------------

@router.post("/title")
async def generate_title(
    req: TitleRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    if chat_engine is None:
        raise HTTPException(status_code=503, detail="AI Service unavailable.")
        
    prompt = (
          f'Generate a concise chat title using EXACTLY 2-4 words.\n'
          f'User asked: "{req.user_message}"\n'
          f'AI: "{req.ai_response}"'
    )

    MODEL_NAME = "qwen3:0.6b"
    PRESET = "CFIR"
    TEMP = 0.7

    try:
        # Use generate_response and extract only the final_answer
        title_response_dict, _ = await chat_engine.generate_response(
            messages=[{"role": "user", "content": prompt}],
            conversation_id=req.conversation_id,
            model=MODEL_NAME,
            preset=PRESET,
            temperature=TEMP,
            user_id=user_id,
            rag_method="no_method", 
            retrieval_method="local" 
        )
        title_html = title_response_dict.get("final_answer", "New chat title")
    except Exception as e:
        logger.error(f"Title generation error: {e}", exc_info=True)
        title_html = "Error Title"


    clean = re.sub(
        r'<think>[\s\S]*?<\/think>'
        r'|<[^>]*>'
        r'|["\']'
        r'|The chat title could be:\s*'
        r'|A Concise Chat Title Using Exactly 2-4 Words\.\s*',
        "",
        title_html,
        flags=re.IGNORECASE,
    ).strip() or "New chat"

    logger.debug(f"generate_title: clean={clean}")
    return {"title": clean}
