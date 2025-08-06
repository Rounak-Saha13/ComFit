import os
from .client import ChatEngine

# Define the embed model path
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "./models/embedding-model")

# Create a global chat engine instance
chat_engine = ChatEngine()

async def chat_answer(
    messages: list,
    conversation_id: str,
    model: str,
    preset: str,
    temperature: float,
    user_id: str = None,
    rag_method: str = "No Specific RAG Method",
    retrieval_method: str = "local context only"
) -> tuple[str, int]:
    """
    Generate a chat response using the chat engine.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        conversation_id: ID of the conversation
        model: Model name to use
        preset: Preset configuration
        temperature: Temperature setting
        user_id: Optional user ID
        
    Returns:
        Tuple of (response_text, duration_ms)
    """
    print(f"DEBUG: chat_answer called with:")
    print(f"  - messages: {messages}")
    print(f"  - conversation_id: {conversation_id}")
    print(f"  - model: {model}")
    print(f"  - model type: {type(model)}")
    print(f"  - preset: {preset}")
    print(f"  - temperature: {temperature}")
    print(f"  - user_id: {user_id}")
    
    # Convert messages to the format expected by ChatEngine
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # Skip system messages for now
        formatted_messages.append({
            "sender": msg["role"],
            "content": msg["content"]
        })
    
    print(f"DEBUG: Formatted messages: {formatted_messages}")
    
    try:
        # Generate response using the chat engine
        result = await chat_engine.generate_response(
            messages=formatted_messages,
            model=model,
            temperature=temperature,
            preset=preset,
            rag_method=rag_method,
            retrieval_method=retrieval_method
        )
        
        print(f"DEBUG: Chat engine result: {result}")
        return result["result"], result["duration"]
    except Exception as e:
        print(f"DEBUG: Error in chat_answer: {e}")
        raise

# Export the main function and constant
__all__ = ["chat_answer", "EMBED_MODEL_PATH", "ChatEngine"]
