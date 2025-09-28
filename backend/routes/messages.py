# In backend/routes/messages.py

from fastapi import APIRouter, HTTPException, Depends, Header, Body
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import time
import re
from fastapi.concurrency import run_in_threadpool
from dateutil import parser 
from pydantic import ValidationError 

# --- FIXED IMPORTS ---
# Using absolute imports based on project structure (schemas, database, chat_engine are top-level packages in 'backend')
from schemas import Message, MessageUpdate, FeedbackRequest, TitleRequest, History, BranchItem, ChatRequest, MessageCreate
from database import supabase, supabase_auth # Assuming database module exists
from chat_engine.client import ChatEngine
# ---------------------

router = APIRouter()

# Global constants for the Supabase Storage bucket and sender names
SUPABASE_IMAGE_BUCKET = 'comfit_images' 
AI_SENDER = "ai" 
USER_SENDER = "user"

# --- GLOBAL CHAT ENGINE INSTANCE ---
# This block attempts to initialize the ChatEngine for immediate use by the routes.
try:
    # Instantiate the ChatEngine here to make it callable globally
    chat_engine_instance = ChatEngine() 
except Exception as e:
    print(f"WARNING: Could not initialize ChatEngine instance in messages.py. All /chat routes will fail: {e}")
    # Define a mock class for graceful failure if initialization fails
    class MockChatEngine:
        async def generate_response(self, *args, **kwargs):
            return "Error: Chat Engine not initialized due to previous error.", 0, None
    chat_engine_instance = MockChatEngine()


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

# -------------------------------------------------------------
# --- NEW: CHAT ENDPOINT FOR SUBMITTING NEW MESSAGES ---
# -------------------------------------------------------------
@router.post("/chat")
async def handle_new_chat(
    payload: ChatRequest = Body(...), 
    user_id: Optional[str] = Depends(get_current_user)
):
    """
    Handles user input, calls the ChatEngine, gets the AI response (and potential image path),
    saves the transaction, and returns the AI's response.
    """
    if not supabase:
         raise HTTPException(status_code=500, detail="Database client not initialized.")
         
    conversation_id = payload.conversation_id
    
    # Extract the latest user message content from the messages list
    user_message_content = payload.messages[-1].content if payload.messages else "" 
    
    # 1. Insert User Message
    try:
        # Insert the full message data provided by the frontend
        user_msg_data = {
            **payload.messages[-1].model_dump(by_alias=True, exclude_none=True),
            "conversation_id": conversation_id,
            "sender": USER_SENDER,
            "created_at": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        supabase.table("messages").insert([user_msg_data]).execute()
    except Exception as e:
        print(f"Error inserting user message: {e}")
        
    # 2. Call Chat Engine (CRITICAL FIX APPLIED HERE)
    try:
        # Create a minimal context for the bot (latest user message)
        messages_context = [{"content": user_message_content, "sender": USER_SENDER}] 
        
        # FIX: Ensure 3 values are unpacked to avoid the "too many values to unpack" error
        ai_content, duration, image_file_path = await chat_engine_instance.generate_response(
            messages=messages_context,
            conversation_id=conversation_id,
            model=payload.model,
            preset=payload.preset,
            temperature=payload.temperature,
            user_id=user_id,
            rag_method=payload.rag_method,
            retrieval_method=payload.retrieval_method
        )
        
        if ai_content.startswith("Error:") or ai_content.startswith("❌"):
             raise HTTPException(status_code=400, detail=ai_content)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat Engine Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

    # 3. Get Public URL from Supabase Storage (if image path is present)
    ai_image_url = None
    if image_file_path:
        try:
            # Synchronous call to Supabase storage must be run in a threadpool
            def get_url_sync():
                return supabase.storage.from_(SUPABASE_IMAGE_BUCKET).get_public_url(image_file_path)

            public_url_data = await run_in_threadpool(get_url_sync)
            
            if isinstance(public_url_data, str) and public_url_data.startswith("http"):
                ai_image_url = public_url_data
                print(f"DEBUG: Generated Public URL for {image_file_path}: {ai_image_url}")
            else:
                print(f"DEBUG: Supabase returned unexpected data for URL generation: {public_url_data}")

        except Exception as e:
            print(f"Error generating public URL from Supabase Storage: {e}")
            ai_image_url = None 

    # 4. Insert AI Message with the Image URL
    ai_message_id = str(uuid.uuid4())
    try:
        ai_message_data = {
            "id": ai_message_id,
            "conversation_id": conversation_id,
            "sender": AI_SENDER,
            "content": ai_content,
            "image_url": ai_image_url, # <-- THE NEW DATA POINT
            "created_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "thinkingTime": duration, # Mapped to Pydantic 'thinkingTime' alias
            "model": payload.model,
            "preset": payload.preset,
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "strategy": payload.strategy,
            "rag_method": payload.rag_method,
            "retrieval_method": payload.retrieval_method,
        }
        
        # Insert into database
        supabase.table("messages").insert([ai_message_data]).execute()
        
        # Return the AI message data back to the frontend, validated by Pydantic
        return Message(**ai_message_data)

    except ValidationError as e:
        print(f"Pydantic Validation Error on AI Response: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Validation Error on AI Response: {str(e)}")
    except Exception as e:
        print(f"Error inserting AI message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save AI response: {str(e)}")


# -------------------------------------------------------------
# --- MODIFIED: HISTORY ENDPOINT TO INCLUDE image_url ---
# -------------------------------------------------------------
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
            .select("id, conversation_id, sender, content, thinking_time, feedback, model, preset, system_prompt, speculative_decoding, temperature, top_p, strategy, rag_method, retrieval_method, created_at, image_url")\
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
            
            # Create Message object using data from the database
            msg = Message(
                id=msg_data["id"],
                conversation_id=msg_data["conversation_id"],
                sender=msg_data["sender"],
                content=msg_data["content"],
                thinkingTime=thinking_time, # Mapped from DB 'thinking_time' to Pydantic 'thinkingTime' alias
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
                # --- NEW MAPPING ---
                imageUrl=msg_data.get("image_url"), # Mapped from DB 'image_url' to Pydantic 'imageUrl' alias
                # -------------------
                created_at=parser.isoparse(msg_data["created_at"])
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
                
                # CORRECTED LINE: use parser.isoparse() for the created_at field
                created_at_value = msg_json.get("created_at")
                if created_at_value:
                    created_at_dt = parser.isoparse(created_at_value)
                else:
                    created_at_dt = datetime.utcnow()

                msg = Message(
                    id=msg_json["id"],
                    conversation_id=msg_json["conversation_id"],
                    sender=msg_json["sender"],
                    content=msg_json["content"],
                    thinkingTime=thinking_time, # Mapped to Pydantic 'thinkingTime' alias
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
                    # --- NEW MAPPING ---
                    imageUrl=msg_json.get("image_url"), # Mapped to Pydantic 'imageUrl' alias
                    # -------------------
                    created_at=created_at_dt
                )
                branch_messages.append(msg)
            
            # Organize by edit point
            if edit_at_id not in branches_by_edit_id:
                branches_by_edit_id[edit_at_id] = []
            
            branch_item = BranchItem(
                branchId=branch_id if not is_original else None,
                isOriginal=is_original,
                messages=branch_messages
            )
            
            branches_by_edit_id[edit_at_id].append(branch_item)
            
            # Set current branch index (default to original branch if available, otherwise last branch)
            if edit_at_id not in current_branch_index_by_edit_id:
                # Find original branch index
                original_index = next((i for i, b in enumerate(branches_by_edit_id[edit_at_id]) if b.is_original), 0)
                current_branch_index_by_edit_id[edit_at_id] = original_index
            
            # Set active branch
            if is_active:
                active_branch_id = branch_id
        
        # Always set original_messages to the main conversation messages
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
    # ... (existing update_message_feedback function) ...
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
    payload: ChatRequest = Body(...), 
    user_id: Optional[str] = Depends(get_current_user)
):
    """Regenerate an AI message - stateless, returns fresh content only"""
    # ... (existing regenerate_message function) ...
    try:
        # Verify message ownership through conversation
        print(f"DEBUG: Looking for message with ID: {message_id}")
        
        # First try to find the message in the messages table
        msg_result = supabase.table("messages")\
            .select("id, conversation_id, content, created_at, sender")\
            .eq("id", message_id)\
            .in_("sender", ["ai", "assistant"])\
            .execute()
        
        print(f"DEBUG: Messages table query result: {msg_result.data}")
        
        # If not found in messages table, check branches table
        if not msg_result.data:
            print(f"DEBUG: Message not found in messages table, checking branches table")
            
            # Get conversation_id from the request payload to narrow down the search
            request_conversation_id = payload.conversation_id
            print(f"DEBUG: Looking for message in branches for conversation: {request_conversation_id}")
            
            branches_result = supabase.table("branches")\
                .select("id, conversation_id, messages, edit_at_id")\
                .eq("conversation_id", request_conversation_id)\
                .execute()
            
            print(f"DEBUG: Found {len(branches_result.data) if branches_result.data else 0} branches for conversation {request_conversation_id}")
            
            # Search through branches for this specific conversation
            found_message = None
            found_conversation_id = None
            found_created_at = None
            
            for branch in branches_result.data:
                branch_messages = branch.get("messages", [])
                print(f"DEBUG: Branch {branch['id']} has {len(branch_messages)} messages")
                for msg in branch_messages:
                    if msg.get("id") == message_id and msg.get("sender") in ["ai", "assistant"]:
                        found_message = msg
                        found_conversation_id = branch["conversation_id"]
                        found_created_at = msg.get("created_at")
                        print(f"DEBUG: Found message in branch {branch['id']}")
                        break
                if found_message:
                    break
            
            if found_message:
                conv_id = found_conversation_id
                original_created_at = found_created_at
                print(f"DEBUG: Found message in branch, conversation: {conv_id}, created at: {original_created_at}")
            else:
                raise HTTPException(status_code=404, detail="AI message not found")
        else:
            conv_id = msg_result.data[0]["conversation_id"]
            original_created_at = msg_result.data[0]["created_at"]
            print(f"DEBUG: Found message in messages table, conversation: {conv_id}, created at: {original_created_at}")
        
        # Verify conversation ownership
        print(f"DEBUG: Verifying ownership of conversation: {conv_id}")
        conv_result = supabase.table("conversations")\
            .select("id")\
            .eq("id", conv_id)\
            .eq("user_id", user_id)\
            .execute()
        
        if not conv_result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        print(f"DEBUG: Conversation ownership verified")
        
        # Get conversation history up to this message
        history_result = supabase.table("messages")\
            .select("*")\
            .eq("conversation_id", conv_id)\
            .lt("created_at", original_created_at)\
            .order("created_at", desc=False)\
            .execute()
        
        # Convert to chat engine format - only include the last user message
        formatted_messages = []
        
        # Find the last user message before the AI message being regenerated
        last_user_message = None
        for msg in reversed(history_result.data):
            if msg["sender"] == "user":
                last_user_message = msg
                break
        
        if last_user_message:
            formatted_messages.append({
                "content": last_user_message["content"],
                "sender": "user" # Use 'sender' to conform to internal bot usage context
            })
        
        # Add system prompt if provided
        if payload.system_prompt:
            formatted_messages.insert(0, {
                "content": payload.system_prompt,
                "sender": "system"
            })
        
        print(f"DEBUG: Sending to chat engine: {formatted_messages}")
        
        # Generate new response using chat engine
        start_time = time.time()
        try:
            # We call generate_response but discard the image_file_path (as it's stateless regen)
            ai_content, duration, _ = await chat_engine_instance.generate_response(
                messages=formatted_messages,
                conversation_id=conv_id,
                model=payload.model,
                preset=payload.preset,
                temperature=payload.temperature,
                user_id=user_id,
                rag_method=payload.rag_method,
                retrieval_method=payload.retrieval_method
            )
            
            if isinstance(ai_content, str) and (ai_content.startswith("Error:") or ai_content.startswith("❌")):
                raise HTTPException(status_code=400, detail=ai_content)
                
        except Exception as e:
            print(f"Chat engine error during regeneration: {e}")
            # Don't save error messages to database - just return the error
            raise HTTPException(status_code=500, detail=f"AI model error during regeneration: {str(e)}")
        
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
            "thinking_time": actual_duration
        }
        
        print(f"DEBUG: Returning response data: {response_data}")
        return response_data
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error regenerating message: {e}")
        # Don't save error messages to database - just return the error
        raise HTTPException(status_code=500, detail=f"Failed to regenerate message: {str(e)}")

@router.post("/title")
async def generate_title(
    req: TitleRequest,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Generates a concise title for a conversation."""
    # ... (existing generate_title function) ...
    prompt = (
        f'Generate a concise chat title using EXACTLY 2-4 words.\n'
        f'User asked: "{req.user_message}"\n'
        f'AI: "{req.ai_response}"'
    )

    MODEL_NAME = "qwen3:0.6b"
    PRESET = "CFIR"
    TEMP = 0.7

    # Generate title using the chat engine. We discard duration and image_file_path
    title_html, _, _ = await chat_engine_instance.generate_response(
        messages=[{"content": prompt, "sender": USER_SENDER}],
        conversation_id=req.conversation_id,
        model=MODEL_NAME,
        preset=PRESET,
        temperature=TEMP,
        user_id=user_id,
        rag_method="no-workflow", 
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
