import os
import time
import uuid
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import asyncio

# import functions from hybrid bot and rag methods
from .Hybrid_Bot import process_mcp_query, MCPRequest, MCPResponse
from .RAG_methods import (
    process_model_context_query,
    run_planning_workflow,
    run_multi_step_query_engine_workflow,
    run_multi_strategy_workflow
)
from .document_manager import document_manager

load_dotenv()

class ChatEngine:
    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "192.168.0.240:11434")
        print(f"DEBUG: ChatEngine initialized with ollama_base_url: {self.ollama_base_url}")
        self.default_model = "llama3:latest"
        
        # Initialize document manager
        self.document_manager = document_manager
        
    def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model name is available in Ollama
        """
        try:
            import ollama
            ollama.show(model_name)
            return True
        except Exception:
            return False
    
    def _get_local_query_engine(self, preset: str = "default"):
        """
        Get a local query engine based on the preset.
        
        Args:
            preset: Preset name (corresponds to index name)
            
        Returns:
            Query engine instance or None if not available
        """
        try:
            # First try to load from document manager (new structure)
            index = self.document_manager.load_index(preset)
            if index:
                from llama_index.llms.ollama import Ollama
                llm = Ollama(model=self.default_model, request_timeout=600.0)
                return index.as_query_engine(llm=llm)
            
            # If not found in document manager, try to load from root vector_store directory
            from llama_index.core import StorageContext, load_index_from_storage
            from pathlib import Path
            
            # vector store directory, change based on where the vector store is stored
            root_vector_store = Path("../vector_store")  
            if root_vector_store.exists():
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=str(root_vector_store))
                    index = load_index_from_storage(storage_context)
                    if index:
                        from llama_index.llms.ollama import Ollama
                        llm = Ollama(model=self.default_model, request_timeout=600.0)
                        print(f"DEBUG: Loaded existing vector store from root directory")
                        return index.as_query_engine(llm=llm)
                except Exception as e:
                    print(f"DEBUG: Could not load root vector store: {e}")
            
            print(f"DEBUG: No index found for preset: {preset}")
            return None
            
        except Exception as e:
            print(f"DEBUG: Error loading query engine for preset {preset}: {e}")
            return None
        
    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system_prompt: str = "You are a helpful AI assistant for comfort and fitting clothing",
        temperature: float = 0.7,
        top_p: float = 0.9,
        rag_method: str = "No Specific RAG Method",
        retrieval_method: str = "local context only",
        preset: str = "CFIR"
    ) -> Dict[str, Any]:
        """
        Generate AI response using the selected model and RAG method
        """
        print(f"DEBUG: ChatEngine.generate_response called with:")
        print(f"  - messages: {messages}")
        print(f"  - model: {model}")
        print(f"  - model type: {type(model)}")
        print(f"  - system_prompt: {system_prompt}")
        print(f"  - temperature: {temperature}")
        print(f"  - top_p: {top_p}")
        print(f"  - rag_method: {rag_method}")
        print(f"  - retrieval_method: {retrieval_method}")
        print(f"  - preset: {preset}")
        print(f"  - ollama_base_url: {self.ollama_base_url}")
        
        start_time = time.time()
        
        # Extract the user's latest message
        user_message = messages[-1]["content"] if messages else ""
        print(f"DEBUG: User message: {user_message}")
        
        # Prepare context for the AI
        context_memory = {
            "conversation_history": messages[:-1],  # All messages except the latest
            "system_prompt": system_prompt,
            "model_config": {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "rag_method": rag_method,
                "retrieval_method": retrieval_method,
                "preset": preset
            }
        }
        print(f"DEBUG: Context memory: {context_memory}")
        
        # Use the model as provided by the frontend
        print(f"DEBUG: Using model provided by frontend: {model}")
        
        try:
            print(f"DEBUG: Processing with RAG method: {rag_method}")
            print(f"DEBUG: Using retrieval method: {retrieval_method}")
            
            # Choose the appropriate processing method based on RAG method
            if rag_method == "Planning Workflow":
                print("DEBUG: Using Planning Workflow")
                result = await self._process_planning_workflow(user_message, context_memory)
            elif rag_method == "Multi-Step Query Engine":
                print("DEBUG: Using Multi-Step Query Engine")
                result = await self._process_multi_step_query(user_message, context_memory)
            elif rag_method == "Multi-Strategy Workflow":
                print("DEBUG: Using Multi-Strategy Workflow")
                result = await self._process_multi_strategy_workflow(user_message, context_memory)
            elif rag_method == "RAC Enhanced Hybrid RAG":
                print("DEBUG: Using RAC Enhanced Hybrid RAG")
                result = await self._process_rac_enhanced_rag(user_message, context_memory)
            else:
                print("DEBUG: Using Default processing")
                # Default processing without specific RAG method
                result = await self._process_default(user_message, context_memory)
            
            duration = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            print(f"DEBUG: Processing completed in {duration}ms")
            print(f"DEBUG: Result: {result}")
            
            return {
                "result": result,
                "duration": duration,
                "ai_message": {
                    "id": str(uuid.uuid4()),
                    "content": result,
                    "thinking_time": duration
                }
            }
            
        except Exception as e:
            print(f"DEBUG: Exception in generate_response: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            duration = int((time.time() - start_time) * 1000)
            error_message = f"Error generating response: {str(e)}"
            
            return {
                "result": error_message,
                "duration": duration,
                "ai_message": {
                    "id": str(uuid.uuid4()),
                    "content": error_message,
                    "thinking_time": duration
                }
            }
    
    async def _process_default(self, user_message: str, context_memory: Dict[str, Any]) -> str:
        """
        Default processing without specific RAG method - calls Ollama directly
        """
        print(f"DEBUG: _process_default called with user_message: {user_message}")
        
        try:
            import ollama
            import os
            
            os.environ['OLLAMA_HOST'] = self.ollama_base_url
            print(f"DEBUG: Set OLLAMA_HOST to: {self.ollama_base_url}")
            
            # Try to configure Ollama client explicitly
            try:
                import ollama
                # Force reload of ollama module to pick up new environment variable
                import importlib
                importlib.reload(ollama)
                print(f"DEBUG: Reloaded ollama module")
            except Exception as reload_error:
                print(f"DEBUG: Error reloading ollama module: {reload_error}")
            
            # Get model configuration from context
            model_config = context_memory.get("model_config", {})
            print(f"DEBUG: model_config from context: {model_config}")
            model = model_config.get("model")
            if not model:
                print(f"DEBUG: No model found in model_config, this should not happen!")
                model = "llama3:latest"  # This should never be reached
            print(f"DEBUG: Extracted model from model_config: {model}")
            temperature = model_config.get("temperature", 0.7)
            system_prompt = context_memory.get("system_prompt", "You are a helpful AI assistant for comfort and fitting clothing")
            
            # Prepare conversation history
            conversation_history = context_memory.get("conversation_history", [])
            
            # Build messages for Ollama
            messages = []
            
            # Add system message
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add conversation history
            for msg in conversation_history:
                role = "user" if msg["sender"] == "user" else "assistant"
                messages.append({
                    "role": role,
                    "content": msg["content"]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            print(f"DEBUG: Calling Ollama with model: {model}, temperature: {temperature}")
            print(f"DEBUG: Messages being sent to Ollama: {messages}")
            
            # Call Ollama
            print(f"DEBUG: About to call Ollama with model: {model}")
            print(f"DEBUG: OLLAMA_HOST environment variable: {os.environ.get('OLLAMA_HOST')}")
            
            # Try to list available models first
            try:
                available_models = ollama.list()
                print(f"DEBUG: Available models from Ollama: {available_models}")
            except Exception as list_error:
                print(f"DEBUG: Error listing models: {list_error}")
            
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            )
            
            print(f"DEBUG: Ollama response: {response}")
            
            # Extract the response content
            ai_response = response.get("message", {}).get("content", "")
            
            if not ai_response:
                print("DEBUG: Empty response from Ollama, using fallback")
                return f"I apologize, but I couldn't generate a response for: {user_message}. Please try again."
            
            print(f"DEBUG: Generated AI response: {ai_response}")
            return ai_response
            
        except Exception as e:
            print(f"DEBUG: Error calling Ollama in _process_default: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            # Fallback response
            return f"I encountered an error while processing your request: {user_message}. Please try again later."
    
    async def _process_planning_workflow(self, user_message: str, context_memory: Dict[str, Any]) -> str:
        """
        Process using Planning Workflow
        """
        try:
            # Import the RAG methods
            from .RAG_methods import run_planning_workflow
            
            # Get model configuration
            model_config = context_memory.get("model_config", {})
            model = model_config.get("model", "llama3:latest")
            preset = model_config.get("preset", "default")
            retrieval_method = model_config.get("retrieval_method", "local context only")
            
            # Setup LLM and embedding model
            from llama_index.llms.ollama import Ollama
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.core import Settings
            from llama_index.core.agent import ReActAgent
            from llama_index.core.tools import FunctionTool
            
            llm = Ollama(model=model, request_timeout=600.0)
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            
            # Get local query engine from document manager or root vector store
            local_query_engine = self._get_local_query_engine(preset)
            
            # Create tools based on retrieval method
            tools = []
            
            # Add local document tool if available and retrieval method allows it
            if local_query_engine and retrieval_method in ["local context only", "Hybrid context", "Smart retrieval"]:
                def local_doc_function(query: str) -> str:
                    try:
                        response = local_query_engine.query(query)
                        return str(response)
                    except Exception as e:
                        return f"Error querying local documents: {e}"
                
                local_tool = FunctionTool.from_defaults(
                    fn=local_doc_function,
                    name="local_document_search",
                    description="Search through uploaded documents and local knowledge base."
                )
                tools.append(local_tool)
            
            # Add web search tool if retrieval method allows it
            if retrieval_method in ["Web searched context only", "Hybrid context", "Smart retrieval"]:
                from .RAG_methods import GoogleCustomSearchTool
                import os
                
                google_api_key = os.getenv("GOOGLE_API_KEY")
                google_cse_id = os.getenv("GOOGLE_CSE_ID")
                
                if google_api_key and google_cse_id:
                    google_search_tool = GoogleCustomSearchTool(
                        api_key=google_api_key,
                        cse_id=google_cse_id,
                        num_results=3
                    )
                    
                    web_tool = FunctionTool.from_defaults(
                        fn=google_search_tool.search,
                        name="web_search",
                        description="Search the web for current information and general knowledge."
                    )
                    tools.append(web_tool)
                else:
                    print("DEBUG: Google API keys not configured, skipping web search")
            
            # If no tools available, create a fallback tool
            if not tools:
                def fallback_function(query: str) -> str:
                    return f"Planning workflow response for: {query} (no document or web search available)"
                
                fallback_tool = FunctionTool.from_defaults(
                    fn=fallback_function,
                    name="fallback_qa",
                    description="General question answering when no specific tools are available."
                )
                tools.append(fallback_tool)
            
            # Create agent with available tools
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=False,
                max_iterations=10
            )
            
            # Execute planning workflow
            trace = []
            result = await run_planning_workflow(user_message, agent, trace)
            
            print(f"DEBUG: Planning workflow completed with trace: {trace}")
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in planning workflow: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return f"Planning workflow encountered an error: {str(e)}"
    
    async def _process_multi_step_query(self, user_message: str, context_memory: Dict[str, Any]) -> str:
        """
        Process using Multi-Step Query Engine
        """
        try:
            # Import the RAG methods
            from .RAG_methods import run_multi_step_query_engine_workflow
            
            # Get model configuration
            model_config = context_memory.get("model_config", {})
            model = model_config.get("model", "llama3:latest")
            preset = model_config.get("preset", "default")
            retrieval_method = model_config.get("retrieval_method", "local context only")
            
            # Setup LLM and embedding model
            from llama_index.llms.ollama import Ollama
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.core import Settings
            
            llm = Ollama(model=model, request_timeout=600.0)
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            
            # Get local query engine from document manager or root vector store
            local_query_engine = self._get_local_query_engine(preset)
            if not local_query_engine and retrieval_method in ["local context only", "Hybrid context", "Smart retrieval"]:
                return f"Local document search not available for preset: {preset}. Please upload documents first."
            
            # Create Google search tool if retrieval method allows it
            from .RAG_methods import GoogleCustomSearchTool
            import os
            
            google_api_key = os.getenv("GOOGLE_API_KEY")
            google_cse_id = os.getenv("GOOGLE_CSE_ID")
            
            if google_api_key and google_cse_id and retrieval_method in ["Web searched context only", "Hybrid context", "Smart retrieval"]:
                google_search_tool = GoogleCustomSearchTool(
                    api_key=google_api_key,
                    cse_id=google_cse_id,
                    num_results=3
                )
            else:
                # fallback if API keys are not available or retrieval method doesn't allow web search
                class MockSearchTool:
                    def search(self, query: str) -> str:
                        return f"Web search not available for query: {query}"
                google_search_tool = MockSearchTool()
            
            # Execute multi-step query engine workflow
            trace = []
            result = await run_multi_step_query_engine_workflow(
                user_message, local_query_engine, google_search_tool, trace, model
            )
            
            print(f"DEBUG: Multi-step query engine completed with trace: {trace}")
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in multi-step query engine: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return f"Multi-step query engine encountered an error: {str(e)}"
    
    async def _process_multi_strategy_workflow(self, user_message: str, context_memory: Dict[str, Any]) -> str:
        """
        Process using Multi-Strategy Workflow
        """
        try:
            # Import the RAG methods
            from .RAG_methods import run_multi_strategy_workflow
            
            # Get model configuration
            model_config = context_memory.get("model_config", {})
            model = model_config.get("model", "llama3:latest")
            preset = model_config.get("preset", "default")
            retrieval_method = model_config.get("retrieval_method", "local context only")
            
            # Setup LLM and embedding model
            from llama_index.llms.ollama import Ollama
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.core import Settings
            
            llm = Ollama(model=model, request_timeout=600.0)
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            
            # Get local query engine from document manager or root vector store
            local_query_engine = self._get_local_query_engine(preset)
            if not local_query_engine and retrieval_method in ["local context only", "Hybrid context", "Smart retrieval"]:
                return f"Local document search not available for preset: {preset}. Please upload documents first."
            
            # Create Google search tool if retrieval method allows it
            from .RAG_methods import GoogleCustomSearchTool
            import os
            
            google_api_key = os.getenv("GOOGLE_API_KEY")
            google_cse_id = os.getenv("GOOGLE_CSE_ID")
            
            if google_api_key and google_cse_id and retrieval_method in ["Web searched context only", "Hybrid context", "Smart retrieval"]:
                google_search_tool = GoogleCustomSearchTool(
                    api_key=google_api_key,
                    cse_id=google_cse_id,
                    num_results=3
                )
            else:
                # Create a mock search tool if API keys are not available or retrieval method doesn't allow web search
                class MockSearchTool:
                    def search(self, query: str) -> str:
                        return f"Web search not available for query: {query}"
                google_search_tool = MockSearchTool()
            
            # Execute multi-strategy workflow
            trace = []
            result = await run_multi_strategy_workflow(
                user_message, local_query_engine, google_search_tool, trace, model
            )
            
            print(f"DEBUG: Multi-strategy workflow completed with trace: {trace}")
            return result
            
        except Exception as e:
            print(f"DEBUG: Error in multi-strategy workflow: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return f"Multi-strategy workflow encountered an error: {str(e)}"
    
    async def _process_rac_enhanced_rag(self, user_message: str, context_memory: Dict[str, Any]) -> str:
        """
        Process using RAC Enhanced Hybrid RAG
        """
        try:
            # Import the RAG methods
            from .RAG_methods import process_model_context_query
            
            # Get model configuration
            model_config = context_memory.get("model_config", {})
            model = model_config.get("model", "llama3:latest")
            preset = model_config.get("preset", "default")
            retrieval_method = model_config.get("retrieval_method", "local context only")
            
            # Setup LLM and embedding model
            from llama_index.llms.ollama import Ollama
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.core import Settings
            from llama_index.core.agent import ReActAgent
            from llama_index.core.tools import FunctionTool
            
            llm = Ollama(model=model, request_timeout=600.0)
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            
            # Get local query engine from document manager or root vector store
            local_query_engine = self._get_local_query_engine(preset)
            if not local_query_engine and retrieval_method in ["local context only", "Hybrid context", "Smart retrieval"]:
                return f"Local document search not available for preset: {preset}. Please upload documents first."
            
            # Create Google search tool if retrieval method allows it
            from .RAG_methods import GoogleCustomSearchTool, RACCorrector
            import os
            
            google_api_key = os.getenv("GOOGLE_API_KEY")
            google_cse_id = os.getenv("GOOGLE_CSE_ID")
            
            if google_api_key and google_cse_id and retrieval_method in ["Web searched context only", "Hybrid context", "Smart retrieval"]:
                google_search_tool = GoogleCustomSearchTool(
                    api_key=google_api_key,
                    cse_id=google_cse_id,
                    num_results=3
                )
            else:
                # Create a mock search tool if API keys are not available or retrieval method doesn't allow web search
                class MockSearchTool:
                    def search(self, query: str) -> str:
                        return f"Web search not available for query: {query}"
                google_search_tool = MockSearchTool()
            
            # Create RAC corrector
            rac_corrector = RACCorrector(
                llm=llm,
                local_query_engine=local_query_engine,
                Google_Search_tool=google_search_tool
            )
            
            # Create agent with tools based on retrieval method
            tools = []
            
            if local_query_engine and retrieval_method in ["local context only", "Hybrid context", "Smart retrieval"]:
                def local_doc_function(query: str) -> str:
                    try:
                        response = local_query_engine.query(query)
                        return str(response)
                    except Exception as e:
                        return f"Error querying local documents: {e}"
                
                local_tool = FunctionTool.from_defaults(
                    fn=local_doc_function,
                    name="local_document_search",
                    description="Search through uploaded documents and local knowledge base."
                )
                tools.append(local_tool)
            
            if google_api_key and google_cse_id and retrieval_method in ["Web searched context only", "Hybrid context", "Smart retrieval"]:
                web_tool = FunctionTool.from_defaults(
                    fn=google_search_tool.search,
                    name="web_search",
                    description="Search the web for current information and general knowledge."
                )
                tools.append(web_tool)
            
            # If no tools available, create a fallback tool
            if not tools:
                def fallback_function(query: str) -> str:
                    return f"RAC enhanced response for: {query} (no document or web search available)"
                
                fallback_tool = FunctionTool.from_defaults(
                    fn=fallback_function,
                    name="fallback_qa",
                    description="General question answering when no specific tools are available."
                )
                tools.append(fallback_tool)
            
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=False,
                max_iterations=10
            )
            
            # Execute RAC enhanced processing
            result = await process_model_context_query(
                query=user_message,
                context_memory=context_memory,
                tool_outputs=[],
                scratchpad="",
                agent_instance=agent,
                rac_corrector_instance=rac_corrector,
                testing_mode=False,
                suppress_threshold=0.4,
                flag_threshold=0.6,
                selected_rag_strategy="rac_enhanced_hybrid_rag",
                local_query_engine=local_query_engine,
                google_custom_search_instance=google_search_tool
            )
            
            print(f"DEBUG: RAC enhanced RAG completed with confidence: {result.get('confidence_score', 0.0)}")
            return result.get("final_answer", "RAC enhanced processing failed")
            
        except Exception as e:
            print(f"DEBUG: Error in RAC enhanced RAG: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return f"RAC enhanced RAG encountered an error: {str(e)}"
    
    async def regenerate_response(
        self,
        message_id: str,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        model: str,
        system_prompt: str = "You are a helpful AI assistant for comfort and fitting clothing",
        temperature: float = 0.7,
        top_p: float = 0.9,
        rag_method: str = "No Specific RAG Method",
        retrieval_method: str = "local context only",
        preset: str = "CFIR"
    ) -> Dict[str, Any]:
        """
        Regenerate AI response for a specific message
        """
        # Find the message to regenerate (remove it and regenerate)
        messages_to_process = [msg for msg in messages if msg["id"] != message_id]
        
        return await self.generate_response(
            messages=messages_to_process,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            rag_method=rag_method,
            retrieval_method=retrieval_method,
            preset=preset
        )

    def list_available_vector_stores(self) -> Dict[str, List[str]]:
        """
        List all available vector stores from both document manager and root directory.
        
        Returns:
            Dictionary with 'document_manager' and 'root_directory' keys containing lists of available indexes
        """
        available_stores = {
            "document_manager": [],
            "root_directory": []
        }
        
        # Get indexes from document manager
        try:
            available_stores["document_manager"] = self.document_manager.list_indexes()
        except Exception as e:
            print(f"DEBUG: Error listing document manager indexes: {e}")
        
        # Check root level vector store directory
        try:
            from pathlib import Path
            root_vector_store = Path("../vector_store")  # Relative to backend directory
            if root_vector_store.exists():
                # Check if there are any index files
                if any(root_vector_store.glob("*.json")):
                    available_stores["root_directory"] = ["default"]
                    print(f"DEBUG: Found existing vector store in root directory")
        except Exception as e:
            print(f"DEBUG: Error checking root vector store: {e}")
        
        return available_stores
    
    def get_available_presets(self) -> List[str]:
        """
        Get all available presets (indexes) that can be used.
        
        Returns:
            List of available preset names
        """
        stores = self.list_available_vector_stores()
        presets = []
        
        # Add document manager presets
        presets.extend(stores["document_manager"])
        
        # Add root directory presets (if any)
        if stores["root_directory"]:
            presets.extend(stores["root_directory"])
        
        # Remove duplicates and return
        return list(set(presets))
    