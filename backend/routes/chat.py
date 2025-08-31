from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
import time, uuid
from datetime import datetime

from schemas import ChatRequest, ChatResponse, Message
from chat_engine import chat_answer
from database import supabase, supabase_auth

router = APIRouter()

def normalize_sender(s: str) -> str:
    s = (s or "").lower()
    if s in ("assistant", "bot", "ai"):
        return "ai"
    return "user" if s == "user" else s

def role_for_llm(sender: str) -> str:
    return "assistant" if normalize_sender(sender) == "ai" else "user"

def ensure_uuid(s: str) -> str:
    try:
        uuid.UUID(str(s))
        return str(s)
    except Exception:
        return str(uuid.uuid4())

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        user = supabase_auth.auth.get_user(token)
        return user.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@router.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        start_time = time.time()

        # Check if this is a branch mode request (has branch_mode flag)
        is_branch_mode = getattr(request, "branch_mode", False)
        
        # 0) Save only the *last* user message once; never overwrite by id.
        # Skip saving to messages table if in branch mode
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
                }
                # INSERT and ignore duplicates (so edits don't overwrite originals)
                try:
                    supabase.table("messages").insert(user_row).execute()
                except Exception as e:
                    # If it's a duplicate-key error, ignore; anything else just log
                    print(f"DEBUG: user insert (ignore duplicates) error: {e}")

        # 1) Build messages for the LLM
        formatted_messages = []
        if request.system_prompt:
            formatted_messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            formatted_messages.append({
                "role": role_for_llm(msg.sender),
                "content": msg.content
            })

        # 2) Call the model
        try:
            ai_content, _duration = await chat_answer(
                messages=formatted_messages,
                conversation_id=request.conversation_id,
                model=request.model,
                preset=request.preset,
                temperature=request.temperature,
                user_id=user_id,
                rag_method=request.rag_method,
                retrieval_method=request.retrieval_method,
            )
            if isinstance(ai_content, str) and ai_content.startswith("Error:"):
                raise HTTPException(status_code=400, detail=ai_content)
        except Exception as e:
            print(f"Chat engine error: {e}")
            # Don't save error messages to database - just return the error
            raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

        actual_duration = int((time.time() - start_time) * 1000)

        # 3) Insert AI message (skip if in branch mode)
        ai_message_id = str(uuid.uuid4())
        created_at_str = datetime.utcnow().isoformat()
        
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
            }
            inserted = supabase.table("messages").insert(ai_row).execute()
            created_at_str = inserted.data[0]["created_at"] if inserted and inserted.data else created_at_str

        # 4) Respond
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
            created_at=datetime.fromisoformat(created_at_str.replace("Z", "+00:00")) if "Z" in created_at_str else datetime.fromisoformat(created_at_str),
        )

        return ChatResponse(
            result=ai_content,
            duration=actual_duration,
            ai_message=ai_message
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        # Don't save error messages to database - just return the error
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")