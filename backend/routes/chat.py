import uuid, re
from fastapi import APIRouter, Depends, HTTPException, Request
from schemas import ChatRequest, ChatResponse, TitleRequest
from chat_engine import chat_answer
from auth import optional_verify_token
from database import supabase
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import ollama
from fastapi.encoders import jsonable_encoder
import os

router = APIRouter()

_guest_logs: dict[str, list[datetime]] = defaultdict(list)

def check_guest_rate_limit(request: Request, user_id: Optional[str]):
    if user_id is not None:
        return
    ip = request.client.host
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=1)
    logs = [ts for ts in _guest_logs[ip] if ts >= cutoff]
    _guest_logs[ip] = logs
    if len(logs) >= 3:
        raise HTTPException(429, "Guests limited to 3 messages per hour")
    _guest_logs[ip].append(now)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    user_id: Optional[str] = Depends(optional_verify_token)
):
    print(f"DEBUG: Chat endpoint called with request: {req}")
    print(f"DEBUG: User ID: {user_id}")
    print(f"DEBUG: Request headers: {dict(request.headers)}")
    print(f"DEBUG: Request client: {request.client}")
    print(f"DEBUG: Model from request: {req.model}")
    print(f"DEBUG: Model type: {type(req.model)}")
    
    try:
        # TODO (fix later): Temporarily disable rate limiting for testing
        # check_guest_rate_limit(request, user_id)
        print("DEBUG: Rate limit check bypassed for testing")
    except Exception as e:
        print(f"DEBUG: Rate limit check failed: {e}")
        raise
    user_msg_id = str(uuid.uuid4())
    user_row = {
        "id": user_msg_id,
        "conversation_id": req.conversation_id,
        "sender": "user",
        "content": req.messages[-1].content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # only persist for real users
    if user_id and supabase:
        try:
            supabase.table("messages").insert(user_row).execute()
            print("DEBUG: User message saved to database")
        except Exception as e:
            print(f"DEBUG: Failed to save user message to database: {e}")
            # Continue without saving - this is not critical for the response

    # prepare messages for LLM
    messages_for_llm = [
        {"role": "system",  "content": req.system_prompt},
        *[{"role": m.sender, "content": m.content} for m in req.messages]
    ]

    # run LLM call directly since chat_answer is async
    print("DEBUG: About to call chat_answer")
    print(f"DEBUG: RAG method from request: {req.rag_method}")
    print(f"DEBUG: Retrieval method from request: {req.retrieval_method}")
    try:
        ai_text, duration = await chat_answer(
            messages_for_llm,
            req.conversation_id,
            req.model,
            req.preset,
            req.temperature,
            user_id,
            req.rag_method,
            req.retrieval_method
        )
        print(f"DEBUG: chat_answer completed successfully: {ai_text[:100]}...")
    except Exception as e:
        print(f"DEBUG: Error in chat_answer: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise

    # store AI response
    ai_msg_id = str(uuid.uuid4())
    ai_row = {
        "id": ai_msg_id,
        "conversation_id": req.conversation_id,
        "sender": "ai",  # Database expects "ai", not "assistant"
        "content": ai_text,
        "thinking_time": duration,
        "model": req.model,
        "preset": req.preset,
        "system_prompt": req.system_prompt,
        "speculative_decoding": req.speculative_decoding,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "strategy": req.strategy,
        "rag_method": req.rag_method,
        "retrieval_method": req.retrieval_method,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # return or persist
    if not user_id or not supabase:
        print("DEBUG: Returning response without database persistence (guest user or no database)")
        # Add created_at field for guest responses
        ai_row["created_at"] = datetime.now(timezone.utc).isoformat()
        # Convert "ai" sender back to "assistant" for frontend compatibility
        ai_row["sender"] = "assistant"
        return {"result": ai_text, "duration": duration, "ai_message": ai_row}

    try:
        print("DEBUG: Attempting to save AI message to database")
        insert_resp = supabase.table("messages").insert(ai_row).execute()
        if not insert_resp.data:
            print("DEBUG: Database insert failed - no data returned")
            # Return response without database record
            return {"result": ai_text, "duration": duration, "ai_message": ai_row}
        ai_record = insert_resp.data[0]
        print("DEBUG: AI message saved to database successfully")
        print("DEBUG: AI record from database:", ai_record)
        print("DEBUG: AI record thinking_time:", ai_record.get("thinking_time"))
    except Exception as e:
        print(f"DEBUG: Database error saving AI message: {e}")
        
        # Check for specific constraint violation
        if "messages_sender_check" in str(e):
            print("DEBUG: Constraint violation - messages_sender_check constraint prevents 'assistant' sender")
            print("DEBUG: This is a database schema issue that needs to be fixed in Supabase dashboard")
            print("DEBUG: Returning response without database record")
            
            # Convert field names for Pydantic model
            ai_message = ai_row.copy()
            if "thinking_time" in ai_message:
                ai_message["thinkingTime"] = ai_message.pop("thinking_time")
            # Convert "ai" sender back to "assistant" for frontend compatibility
            if "sender" in ai_message and ai_message["sender"] == "ai":
                ai_message["sender"] = "assistant"
            return {"result": ai_text, "duration": duration, "ai_message": ai_message}
        
        elif "duplicate key" in str(e).lower():
            print("DEBUG: Duplicate key error - returning response without database record")
            # Convert field names for Pydantic model
            ai_message = ai_row.copy()
            if "thinking_time" in ai_message:
                ai_message["thinkingTime"] = ai_message.pop("thinking_time")
            # Convert "ai" sender back to "assistant" for frontend compatibility
            if "sender" in ai_message and ai_message["sender"] == "ai":
                ai_message["sender"] = "assistant"
            return {"result": ai_text, "duration": duration, "ai_message": ai_message}
        
        # For other database errors, return response without database record
        print("DEBUG: Database error - returning response without database record")
        # Convert field names for Pydantic model
        ai_message = ai_row.copy()
        if "thinking_time" in ai_message:
            ai_message["thinkingTime"] = ai_message.pop("thinking_time")
        # Convert "ai" sender back to "assistant" for frontend compatibility
        if "sender" in ai_message and ai_message["sender"] == "ai":
            ai_message["sender"] = "assistant"
        return {"result": ai_text, "duration": duration, "ai_message": ai_message}

    print("DEBUG: Final response ai_message:", ai_record)
    
    # Ensure thinking_time is included in the response
    if "thinking_time" not in ai_record:
        print("DEBUG: thinking_time not found in ai_record, adding it")
        ai_record["thinking_time"] = duration
    
    # Convert database field names to Pydantic model field names
    if "thinking_time" in ai_record:
        ai_record["thinkingTime"] = ai_record.pop("thinking_time")
    
    # Convert "ai" sender back to "assistant" for frontend compatibility
    if "sender" in ai_record and ai_record["sender"] == "ai":
        ai_record["sender"] = "assistant"
    
    return {"result": ai_text, "duration": duration, "ai_message": ai_record}

@router.get("/models", tags=["models"])
async def list_models():
    print("DEBUG: Models endpoint called")
    
    try:
        print("DEBUG: Attempting to call ollama.list()")
        
        # Set the OLLAMA_HOST environment variable to use the same remote instance
        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        os.environ['OLLAMA_HOST'] = ollama_host
        print(f"DEBUG: Using OLLAMA_HOST: {ollama_host}")
        
        # Reload ollama module to pick up the new environment variable
        import importlib
        importlib.reload(ollama)
        
        list_response = ollama.list()
        print(f"DEBUG: Ollama response: {list_response}")
        data = jsonable_encoder(list_response)
        print(f"DEBUG: Encoded data: {data}")
    except Exception as e:
        print(f"DEBUG: Error calling ollama.list(): {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {e}",
        )

    # get model names
    models = [entry.get("model") for entry in data.get("models", []) if entry.get("model")]
    if "nomic-embed-text:latest" in models:
        models.remove("nomic-embed-text:latest")
    print("DEBUG: Extracted models:", models)
    return {"models": models}

@router.get("/presets", tags=["chat"])
async def list_available_presets():
    """
    List all available presets (vector stores) that can be used for RAG operations.
    """
    print("DEBUG: Presets endpoint called")
    try:
        from chat_engine import chat_engine
        presets = chat_engine.get_available_presets()
        print(f"DEBUG: Available presets: {presets}")
        return {"presets": presets}
    except Exception as e:
        print(f"DEBUG: Error listing presets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list presets: {e}",
        )

@router.post("/title")
async def generate_title(
    req: TitleRequest,
    user_id: Optional[str] = Depends(optional_verify_token)
):
    prompt = (
         f'Generate a concise chat title using EXACTLY 2-4 words. Do not include more than 5 words.\n'
         f'User asked: "{req.user_message}"\n'
         f'AI: "{req.ai_response}"'
    )

    MODEL_NAME = "qwen3:0.6b"
    PRESET = "CFIR"
    TEMP = 0.7

    # run title generation directly since chat_answer is async
    title_html, _ = await chat_answer(
        [{"role": "user", "content": prompt}],
        req.conversation_id,
        MODEL_NAME,
        PRESET,
        TEMP,
        user_id
    )

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

    print(f"DEBUG: generate_title: clean={clean}")
    return {"title": clean}

@router.post("/debug-rag", tags=["chat"])
async def debug_rag_methods(
    req: ChatRequest,
    request: Request,
    user_id: Optional[str] = Depends(optional_verify_token)
):
    """
    Debug endpoint to test different RAG methods and see their behavior.
    """
    print(f"DEBUG: Debug RAG endpoint called with:")
    print(f"  - rag_method: {req.rag_method}")
    print(f"  - retrieval_method: {req.retrieval_method}")
    print(f"  - preset: {req.preset}")
    print(f"  - model: {req.model}")
    
    try:
        # Check available presets
        from chat_engine import chat_engine
        available_presets = chat_engine.get_available_presets()
        print(f"DEBUG: Available presets: {available_presets}")
        
        # Check if the requested preset is available
        preset_available = req.preset in available_presets
        print(f"DEBUG: Preset '{req.preset}' available: {preset_available}")
        
        # Check if local query engine is available
        local_query_engine = chat_engine._get_local_query_engine(req.preset)
        local_available = local_query_engine is not None
        print(f"DEBUG: Local query engine available: {local_available}")
        
        # Check Google API keys
        import os
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        web_search_available = bool(google_api_key and google_cse_id)
        print(f"DEBUG: Web search available: {web_search_available}")
        
        # Determine what tools will be used based on retrieval method
        tools_analysis = {
            "rag_method": req.rag_method,
            "retrieval_method": req.retrieval_method,
            "preset": req.preset,
            "preset_available": preset_available,
            "local_query_engine_available": local_available,
            "web_search_available": web_search_available,
            "will_use_local": req.retrieval_method in ["local context only", "Hybrid context", "Smart retrieval"] and local_available,
            "will_use_web": req.retrieval_method in ["Web searched context only", "Hybrid context", "Smart retrieval"] and web_search_available,
            "available_presets": available_presets
        }
        
        return {
            "debug_info": tools_analysis,
            "message": "RAG method analysis completed. Check the debug info to see what tools will be used."
        }
        
    except Exception as e:
        print(f"DEBUG: Error in debug RAG endpoint: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Debug failed: {e}",
        )