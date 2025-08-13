from fastapi import APIRouter, HTTPException, Depends, Header, Body
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import time
import re
from fastapi.concurrency import run_in_threadpool
from schemas import Message, MessageUpdate, FeedbackRequest, TitleRequest, History, BranchItem
from chat_engine import chat_answer
from database import supabase, supabase_auth

router = APIRouter()

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

@router.get("/messages/conversation/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get conversation history with branches"""
    try:
        print(f"DEBUG: Loading history for conversation {conversation_id}")
        
        # Verify conversation ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conversation_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get all messages for this conversation (main conversation flow)
        messages_result = supabase.table("messages")\
            .select("id, conversation_id, sender, content, thinking_time, feedback, model, preset, system_prompt, speculative_decoding, temperature, top_p, strategy, rag_method, retrieval_method, created_at")\
            .eq("conversation_id", conversation_id)\
            .order("created_at", desc=False)\
            .execute()
        
        print(f"DEBUG: Found {len(messages_result.data) if messages_result.data else 0} main messages")
        
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
        
        print(f"DEBUG: Found {len(branches_result.data) if branches_result.data else 0} branches")
        
        # Organize messages and branches
        messages = []
        original_messages = []
        branches_by_edit_id = {}
        current_branch_index_by_edit_id = {}
        active_branch_id = None
        
        # Process main conversation messages
        for msg_data in messages_result.data:
            # Handle missing thinking_time field gracefully
            thinking_time = msg_data.get("thinking_time")
            if thinking_time is None or thinking_time == "":
                thinking_time = 0
            else:
                try:
                    thinking_time = int(thinking_time)
                except (ValueError, TypeError):
                    thinking_time = 0
            
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
                created_at=datetime.fromisoformat(msg_data["created_at"])
            )
            messages.append(msg)
        
        # Process branches - use JSON messages stored in branches table
        for branch_data in branches_result.data:
            edit_at_id = branch_data["edit_at_id"]
            branch_id = branch_data["id"]
            is_original = branch_data["is_original"]
            is_active = branch_data["is_active"]
            
            print(f"DEBUG: Processing branch {branch_id} (edit_at: {edit_at_id}, is_original: {is_original}, is_active: {is_active})")
            
            # Get messages directly from branch JSON (not from messages table)
            branch_messages_json = branch_data.get("messages", [])
            branch_messages = []
            
            print(f"DEBUG: Branch {branch_id} has {len(branch_messages_json)} JSON messages")
            
            # Convert JSON messages to Message objects
            for msg_json in branch_messages_json:
                # Handle missing thinking_time field gracefully
                thinking_time = msg_json.get("thinking_time")
                if thinking_time is None or thinking_time == "":
                    thinking_time = 0
                else:
                    try:
                        thinking_time = int(thinking_time)
                    except (ValueError, TypeError):
                        thinking_time = 0
                
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
                    created_at=datetime.fromisoformat(msg_json["created_at"]) if msg_json.get("created_at") else datetime.utcnow()
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
            
            # Set current branch index (default to last branch)
            if edit_at_id not in current_branch_index_by_edit_id:
                current_branch_index_by_edit_id[edit_at_id] = len(branches_by_edit_id[edit_at_id]) - 1
            
            # Set active branch
            if is_active:
                active_branch_id = branch_id
        
        # Always set original_messages to the main conversation messages
        # This represents the original conversation flow before any branching
        original_messages = messages.copy()
        
        print(f"DEBUG: Final structure - messages: {len(messages)}, original: {len(original_messages)}, branches: {len(branches_by_edit_id)}")
        
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
        print(f"Error getting conversation history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")

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
        print(f"Error updating feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update feedback: {str(e)}")

@router.post("/messages/{message_id}/regenerate")
async def regenerate_message(
    message_id: str,
    request: dict = Body(...),
    user_id: str = Depends(get_current_user)
):
    """Regenerate an AI message"""
    try:
        # Verify message ownership through conversation
        msg_result = supabase.table("messages")\
            .select("id, conversation_id, content")\
            .eq("id", message_id)\
            .eq("sender", "ai")\
            .execute()
        
        if not msg_result.data:
            raise HTTPException(status_code=404, detail="AI message not found")
        
        conv_id = msg_result.data[0]["conversation_id"]
        
        # Verify conversation ownership
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conv_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get conversation history up to this message
        history_result = supabase.table("messages")\
            .select("*")\
            .eq("conversation_id", conv_id)\
            .lt("created_at", msg_result.data[0]["created_at"])\
            .order("created_at", desc=False)\
            .execute()
        
        # Convert to chat engine format
        formatted_messages = []
        for msg in history_result.data:
            formatted_messages.append({
                "role": msg["sender"],
                "content": msg["content"]
            })
        
        # Add system prompt if provided
        if request.get("system_prompt"):
            formatted_messages.insert(0, {
                "role": "system",
                "content": request["system_prompt"]
            })
        
        # Generate new response
        start_time = time.time()
        ai_content, duration = await chat_answer(
            messages=formatted_messages,
            conversation_id=conv_id,
            model=request.get("model", "gemma3:latest"),
            preset=request.get("preset", "CFIR"),
            temperature=request.get("temperature", 0.7),
            user_id=user_id,
            rag_method=request.get("rag_method", "No Specific RAG Method"),
            retrieval_method=request.get("retrieval_method", "local context only")
        )
        
        actual_duration = int((time.time() - start_time) * 1000)
        
        # Update the message
        result = supabase.table("messages")\
            .update({
                "content": ai_content,
                "thinking_time": actual_duration,
                "model": request.get("model"),
                "preset": request.get("preset"),
                "temperature": request.get("temperature"),
                "rag_method": request.get("rag_method"),
                "retrieval_method": request.get("retrieval_method"),
                "updated_at": datetime.utcnow().isoformat()
            })\
            .eq("id", message_id)\
            .execute()
        row = result.data[0]
        if result.data:
            updated_msg = Message(
                id=message_id,
                conversation_id=conv_id,
                sender="ai",
                content=ai_content,
                thinking_time=actual_duration,
                feedback=None,
                model=request.get("model"),
                preset=request.get("preset"),
                temperature=request.get("temperature"),
                rag_method=request.get("rag_method"),
                retrieval_method=request.get("retrieval_method"),
                created_at=datetime.fromisoformat(row["created_at"])
            )
            return updated_msg
        else:
            raise HTTPException(status_code=500, detail="Failed to update message")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error regenerating message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate message: {str(e)}")

@router.post("/title")
async def generate_title(
    req: TitleRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    prompt = (
         f'Generate a concise chat title using EXACTLY 2-4 words.\n'
         f'User asked: "{req.user_message}"\n'
         f'AI: "{req.ai_response}"'
    )

    MODEL_NAME = "qwen3:0.6b"
    PRESET = "CFIR"
    TEMP = 0.7

    # Generate title using the chat engine
    title_html, _ = await chat_answer(
        messages=[{"role": "user", "content": prompt}],
        conversation_id=req.conversation_id,
        model=MODEL_NAME,
        preset=PRESET,
        temperature=TEMP,
        user_id=user_id,
        rag_method="no-workflow",  # Use default strategy for title generation
        retrieval_method="local context only"
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