import os
import sys
import time
import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv


# Import core pieces from your Hybrid_Bot module
from .Hybrid_Bot import (
    process_model_context_query,
    RACCorrector,
    ReActAgent,
    GoogleCustomSearchTool,
    VectorStoreIndex,
    Ollama,
    OllamaEmbedding,
    Settings,
    FunctionTool,
    validate_google_api_keys_from_env,
    load_documents_for_indexing,
    _extract_source_filenames,
)


from .document_manager import document_manager


# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (client.py) - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------
# Normalization helpers
# -----------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


RAG_STRATEGY_MAP = {
    # canonical
    "rac_enhanced_hybrid_rag": "rac_enhanced_hybrid_rag",
    "planning_workflow": "planning_workflow",
    "multi_step_query_engine": "multi_step_query_engine",
    "multi_strategy_workflow": "multi_strategy_workflow",
    "no_method": "no_method",
    # common labels / UI strings
    "no specific rag method": "no_method",
    "no specific method": "no_method",
    "none": "no_method",
    "default": "rac_enhanced_hybrid_rag",
}


RETRIEVAL_METHOD_MAP = {
    # canonical
    "local": "local",
    "web": "web",
    "hybrid": "hybrid",
    "automatic": "automatic",
    # common labels / UI strings
    "local context only": "local",
    "local only (pdf)": "local",
    "web only": "web",
    "internet": "web",
}


DEFAULT_RAG_STRATEGY = "multi_step_query_engine"
DEFAULT_RETRIEVAL = "hybrid"


# Define the structured output format for the client.py's public method
ClientResponse = Dict[str, Any] # Will contain final_answer, image_url, sources_str, duration


# -----------------------
# Chat Engine
# -----------------------
class ChatEngine:
    def __init__(self):
        """
        Initializes all necessary components from Hybrid_Bot.py.
        Runs once on app startup.
        """
        logger.info("Initializing core components from hybrid_bot.py...")


        load_dotenv()


        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = "llama3"
        self.local_query_engine = None
        self.google_search_instance = None
        self.rac_corrector = None
        self.agent = None
        self.tools_for_agent: List[FunctionTool] = []


        # Determine testing mode from command line arguments
        self.testing_mode_enabled = "--dry-run" in sys.argv
        if self.testing_mode_enabled:
            sys.argv.remove("--dry-run")


        # LlamaIndex global settings
        Settings.llm = Ollama(model=self.default_model, request_timeout=600.0, base_url=self.ollama_base_url)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


        try:
            # 1) Load & index local documents
            documents = load_documents_for_indexing()
            logger.info("Creating VectorStoreIndex for local PDF data...")
            self.local_index = VectorStoreIndex.from_documents(
                documents,
                llm=Settings.llm,
                embed_model=Settings.embed_model
            )
            self.local_query_engine = self.local_index.as_query_engine(
                llm=Settings.llm,
                response_mode="tree_summarize",
                similarity_top_k=5,
            )
            logger.info("Local documents indexed and query engine created.")


            # 2) Google Search Tool
            google_api_key, google_cse_id = validate_google_api_keys_from_env()
            # NOTE: We allow initialization even without keys, to enable local-only mode.
            if not (google_api_key and google_cse_id):
                 logger.warning("Google API keys not configured. Web search will be disabled if requested.")
            
            self.google_search_instance = GoogleCustomSearchTool(
                api_key=google_api_key or "", # Pass empty strings if None
                cse_id=google_cse_id or "",
                num_results=5
            )
            logger.info("Google Search tool initialized.")


            # 3) RAC Corrector
            self.rac_corrector = RACCorrector(
                llm=Settings.llm,
                local_query_engine=self.local_query_engine,
                Google_Search_tool=self.google_search_instance
            )
            self.rac_corrector.testing_mode = self.testing_mode_enabled
            logger.info("RAC Corrector initialized.")


            # 4) Tools for agent
            # Wrap local query to be resilient
            def _local_book_qa_function(q: str) -> str:
                try:
                    resp_obj = self.local_query_engine.query(q)
                    text = str(resp_obj)
                    # Use the robust extraction function from Hybrid_Bot.py
                    files = _extract_source_filenames(resp_obj)
                    if files:
                        text += "\n\nLocal Sources: " + ", ".join(sorted(set(files)))
                    return text
                except Exception as e:
                    logger.warning(f"Local RAG tool error: {e}")
                    return "Local search failed for this query."


            # Define the tools (Image tool will be dynamically added later)
            local_rag_tool = FunctionTool.from_defaults(
                fn=_local_book_qa_function,
                name="local_book_qa",
                description="Useful for questions specifically about the content of the provided PDF documents."
            )


            google_search_tool_for_agent = FunctionTool.from_defaults(
                fn=self.google_search_instance.search_legacy,
                name="google_web_search",
                description="Useful for general knowledge questions, current events, or anything requiring internet search."
            )
            
            # NOTE: We can't import image_rag_tool from Hybrid_Bot.py since it wasn't exposed.
            # We'll just pass the full self.tools_for_agent which will be filtered 
            # by `process_model_context_query` based on the retrieval method.
            # For now, we only initialize the RAG and Web tools.
            self.tools_for_agent = [local_rag_tool, google_search_tool_for_agent]
            # The image_rag_tool is generated internally in main() of Hybrid_Bot.py, 
            # but is now integrated by passing the full list to ReActAgent inside the core function.
            # We trust that the core `process_model_context_query` handles tool filtering/usage correctly.


            # 5) Main ReAct Agent
            # Note: We re-initialize the agent *inside* process_model_context_query 
            # to handle tool filtering per query. This global agent instance is 
            # just a placeholder, but is required by the function signature.
            self.agent = ReActAgent.from_tools(
                llm=Settings.llm,
                tools=self.tools_for_agent, # Pass a base set of tools
                verbose=False,
                max_iterations=30
            )
            logger.info("Main ReAct Agent initialized.")
        except Exception as e:
            logger.critical(f"FATAL ERROR during ChatEngine initialization: {e}", exc_info=True)
            sys.exit(1)


    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
        conversation_id: str,
        model: str,
        preset: str,
        temperature: float,
        user_id: str,
        rag_method: str,
        retrieval_method: str,
    ) -> tuple[ClientResponse, int]: # Updated return type to ClientResponse (Dict)
        """
        Orchestrates the RAG pipeline and returns (ClientResponse, duration_ms).
        ClientResponse is a dictionary containing final_answer, image_url, etc.
        """
        logger.info(f"Received request for conversation_id: {conversation_id}")
        logger.info(f"User ID: {user_id}")


        # Extract the latest user message
        user_query = messages[-1].get("content", "") if messages else ""
        if not user_query or not user_query.strip():
            return {"final_answer": "Error: User query is empty.", "image_url": None, "sources_str": ""}, 0


        # Normalize strategy/retrieval labels -> canonical keys expected by core
        rag_key = RAG_STRATEGY_MAP.get(_norm(rag_method), DEFAULT_RAG_STRATEGY)
        retrieval_key = RETRIEVAL_METHOD_MAP.get(_norm(retrieval_method), DEFAULT_RETRIEVAL)


        start_time = time.time()
        mcp_response: Dict[str, Any] = {} # Initialize mcp_response
        
        try:
            mcp_response = await process_model_context_query(
                query=user_query,
                context_memory=messages,
                tool_outputs=[],
                scratchpad="",
                agent_instance=self.agent, # Base agent instance
                rac_corrector_instance=self.rac_corrector,
                testing_mode=self.testing_mode_enabled,
                suppress_threshold=0.4,
                flag_threshold=0.6,
                selected_rag_strategy=rag_key,
                selected_retrieval_method=retrieval_key,
                local_query_engine=self.local_query_engine,
                google_custom_search_instance=self.google_search_instance,
                tools_for_agent=self.tools_for_agent # Base tool list (will be filtered in core)
            )


            final_answer = mcp_response.get("final_answer", "No answer generated.")
            
            # --- Handle Invalid Strategy Retry ---
            if final_answer.startswith("Invalid RAG strategy selected"):
                logger.warning("Invalid RAG strategy from upstream. Retrying with safe defaults (multi_step_query_engine, hybrid).")


                mcp_response = await process_model_context_query(
                    query=user_query,
                    context_memory=messages,
                    tool_outputs=[],
                    scratchpad="",
                    agent_instance=self.agent,
                    rac_corrector_instance=self.rac_corrector,
                    testing_mode=self.testing_mode_enabled,
                    suppress_threshold=0.4,
                    flag_threshold=0.6,
                    selected_rag_strategy=DEFAULT_RAG_STRATEGY,
                    selected_retrieval_method=DEFAULT_RETRIEVAL,
                    local_query_engine=self.local_query_engine,
                    google_custom_search_instance=self.google_search_instance,
                    tools_for_agent=self.tools_for_agent
                )
                final_answer = mcp_response.get("final_answer", "No answer generated.")
            
            # --- Format Final Response Dictionary ---
            sources_info = mcp_response.get("sources_used", {})
            sources_str = self.format_sources_info(sources_info)
            image_url = mcp_response.get("image_url", None) # <--- CAPTURE NEW FIELD
            
            # Construct the response object
            response_dict = {
                "final_answer": final_answer,
                "image_url": image_url, # Propagate the image URL
                "sources_str": sources_str, # The formatted source text
                "trace": mcp_response.get("trace", []) # Optionally propagate the trace
            }


            duration = int((time.time() - start_time) * 1000)
            return response_dict, duration


        except Exception as e:
            logger.error(f"Error in ChatEngine.generate_response: {e}", exc_info=True)
            duration = int((time.time() - start_time) * 1000)
            return {
                "final_answer": f"Error: An unexpected error occurred while processing your request. Details: {str(e)}",
                "image_url": None,
                "sources_str": "📚 **Sources Used:** ❌ Internal Error",
            }, duration


    def format_sources_info(self, sources_info: Dict[str, Any]) -> str:
        """Formats sources into a user-friendly string."""
        info_lines = ["📚 **Sources Used:**"]
        local_files = sources_info.get('local_files', [])
        web_links = sources_info.get('web_links', [])


        if local_files:
            info_lines.append("  📄 **Local PDF Documents Referenced:**")
            for i, filename in enumerate(local_files, 1):
                info_lines.append(f"    {i}. {filename}")


        if web_links:
            info_lines.append("  🌐 **Web Sources Referenced:**")
            for i, link in enumerate(web_links, 1):
                title = link.get('title', 'Unknown Title')
                url = link.get('url', '')
                info_lines.append(f"    {i}. {title} ({url})")
        
        # NOTE: We do not include image_url here, as it should be kept separate 
        # for frontend rendering, but we ensure one of the sections is shown 
        # if the image tool was successful.
        if not local_files and not web_links and not sources_info.get("image_url"):
            info_lines.append("  ℹ️ No external sources were consulted.")


        return "\n".join(info_lines)



# -----------------------
# Standalone CLI (optional)
# -----------------------
if __name__ == "__main__":
    # Load env and spin up for quick manual testing
    load_dotenv()
    engine = ChatEngine()


    async def main_cli():
        while True:
            user_question = input("\nEnter your question (or 'exit'): ").strip()
            if user_question.lower() == 'exit':
                break


            print("--- Generating Response ---")


            mock_request_params = {
                "messages": [{"content": user_question, "sender": "user"}],
                "conversation_id": str(uuid.uuid4()),
                "model": "llama3",
                "preset": "default",
                "temperature": 0.7,
                "user_id": "test_user_id",
                "rag_method": "No Specific RAG Method",    # UI label; normalization will fix this
                "retrieval_method": "local context only",  # UI label; normalization will fix this
            }


            start_time = time.time()
            # The CLI needs to unpack the structured response
            response_dict, _duration_ms = await engine.generate_response(**mock_request_params)
            end_time = time.time()
            final_answer = response_dict.get("final_answer", "")
            image_url = response_dict.get("image_url")
            sources_str = response_dict.get("sources_str", "")


            print("\n" + "=" * 50)
            print(f"Final Answer (Time: {end_time - start_time:.2f}s):")
            print("=" * 50)
            print(final_answer)
            
            if image_url:
                print(f"\n🖼️ **IMAGE URL:** {image_url}")
                
            print("\n--- Sources ---")
            print(sources_str)
            print("-" * 50)


    asyncio.run(main_cli())