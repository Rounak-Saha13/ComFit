import os
import logging
import sys
import PyPDF2
import re
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize
import hashlib
import traceback
import time
import asyncio
from functools import partial
from pathlib import Path
import shutil
import tempfile

# --- NEW IMPORTS FOR SUPABASE & STRUCTURED OUTPUT ---
from supabase import create_client, Client
from pydantic import BaseModel, Field

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Load environment variables from .env file ---
load_dotenv()

# --- Global Exception Hook for Tracebacks ---
def my_excepthook(type, value, traceback_obj):
    import traceback as tb
    tb.print_exception(type, value, traceback_obj)
    sys.exit(1)
sys.excepthook = my_excepthook

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from LlamaIndex and HTTP client libraries
logging.getLogger('llama_index').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Correcting the import for older LlamaIndex versions
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import MultiStepQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
# Corrected import path for Response for older LlamaIndex versions
from llama_index.core.base.response.schema import Response

# --- Model Context Protocol Prompt Template ---
MODEL_CONTEXT_PROMPT = """
### User Query:
{query}

### Session Context:
{context_memory}

### Tool Outputs:
{tool_outputs}

### Scratchpad (Reasoning History):
{scratchpad}

### Instructions:
You are a helpful assistant. Use the context and available tool outputs to answer the user’s query as clearly and directly as possible.
Avoid mentioning any tools or internal steps. Provide only the final answer.

### Answer:
"""

# --- LLM Setup ---
OPTIMAL_LLM_MODEL_NAME = "llama3"
OPTIMAL_LLM = Ollama(model=OPTIMAL_LLM_MODEL_NAME, request_timeout=600.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = OPTIMAL_LLM

# --- Helper to extract source filenames from LlamaIndex Response object ---
def _extract_source_filenames(response_obj: Response) -> List[str]:
    """Extracts unique filenames from the source nodes of a LlamaIndex Response object."""
    if not hasattr(response_obj, 'source_nodes') or not response_obj.source_nodes:
        return []
    
    unique_filenames = set()
    for node in response_obj.source_nodes:
        # In older versions, metadata might be in a different structure
        filename = node.metadata.get('filename', None)
        if filename:
            unique_filenames.add(filename)
    return sorted(list(unique_filenames))

# --- New Pydantic model for Intent Classification ---
class UserIntent(BaseModel):
    """Structured output for classifying user intent."""
    intent: str = Field(description="The primary action requested: 'image' for visual requests, 'text' for factual/RAG questions, or 'general' for simple chat.")
    query_term: str = Field(description="The key term or topic for the search (e.g., 'anthropometry' if the intent is 'image', or the full question if 'text').")

# --- LLM Prompt for Intent Classification ---
IMAGE_INTENT_PROMPT = """
You are a router. Analyze the user's query and classify the primary intent.
Prioritize 'image' if the user uses keywords like 'show me', 'picture of', 'image of', 'photo of', 'graph of', or 'diagram of'.

Query: "{query}"

Output the result in a JSON object that conforms to the UserIntent schema.
"""

def classify_intent_llm(query: str, llm: Ollama) -> UserIntent:
    """
    Uses the LLM to classify the user's intent and extract the search term.
    """
    logger.info(f"Classifying intent for query: '{query}'")
    
    # Use the LLM's structured output capability
    try:
        response = llm.structured_predict(UserIntent, IMAGE_INTENT_PROMPT.format(query=query))
        return response
    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}. Falling back to default 'text' intent.")
        # Fallback to simple keyword check
        image_keywords = ["show me", "picture of", "image of", "photo of", "graph of", "diagram of"]
        if any(keyword in query.lower() for keyword in image_keywords):
            return UserIntent(intent="image", query_term=query)
        
        return UserIntent(intent="text", query_term=query)

# --- RAC (Retrieval-Augmented Correction) Implementation (OMITTED FOR BREVITY, ASSUMED VALID) ---
# NOTE: The full RAC logic is replaced with placeholders here, but the functions must exist locally.
class FactualClaimExtractor:
    def __init__(self, llm):
        self.llm = llm
    
    def extract_claims(self, text: str) -> List[str]:
        return []

class FactVerifier:
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.local_query_engine = local_query_engine
        self.Google_Search_tool = Google_Search_tool
        self.retrieval_method = retrieval_method 
        self.verification_cache = {} 
        self.reversal_min_confidence = 0.95

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        return {'is_supported': True, 'confidence': 1.0, 'evidence': [], 'web_sources': [], 'local_source_files': [], 'correction_suggestion': None, 'warning': None}
    
    def _extract_search_terms(self, claim: str) -> str:
        return claim 

class RACCorrector:
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.claim_extractor = FactualClaimExtractor(llm)
        self.fact_verifier = FactVerifier(llm, local_query_engine, Google_Search_tool, retrieval_method)
        self.rac_enabled = True 
        self.testing_mode = False
        self.correction_threshold = 0.5 
        self.uncertainty_threshold = 0.6 

    def correct_response(self, original_response: str, apply_corrections: bool = True) -> Dict[str, Any]:
        if not self.rac_enabled:
            return {'corrected_response': original_response, 'claims_analyzed': 0, 'corrections_made': 0, 'verification_results': [], 'uncertain_claims': [], 'average_confidence': 1.0 }
        
        return {
            'original_response': original_response,
            'corrected_response': original_response,
            'claims_analyzed': 1,
            'corrections_made': 0,
            'verification_results': [],
            'uncertain_claims': [],
            'average_confidence': 1.0
        }
    
    def _apply_corrections(self, original_response: str, corrections: List[Dict]) -> str:
        return original_response
# --- END RAC PLACHOLDER ---

# --- Google Search Tool ---
def validate_google_api_keys_from_env():
    """Validates presence of Google API keys from environment variables."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not google_api_key or not google_cse_id:
        logger.error("Google API keys not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID in your .env file.")
        return None, None
    logger.info("Google API keys validated.")
    return google_api_key, google_cse_id

class GoogleCustomSearchTool:
    # ... (GoogleCustomSearchTool implementation from your original code)
    def __init__(self, api_key: str, cse_id: str, num_results: int = 3):
        self.api_key = api_key
        self.cse_id = cse_id
        self.num_results = num_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> tuple[str, list]:
        # Placeholder for actual web search logic
        return "Web search is simulated.", []

    def search_legacy(self, query: str) -> str:
        result, _ = self.search(query)
        return result

# --- NEW: Supabase Image Retrieval Tool ---
class SupabaseImageTool:
    """
    Handles image retrieval from the Supabase database based on a user query.
    """
    def __init__(self, supabase_url: str, supabase_key: str):
        # Initializes the Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = "images"  # Your image table name

    def retrieve_image_urls(self, search_term: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Searches the 'images' table for a matching prompt and returns the public URL.
        """
        logger.info(f"Supabase Image Search: Looking up '{search_term}'...")
        
        # Use ILIKE with wildcards to search for the search_term anywhere within the 'prompt' column
        safe_search_term = f"%{search_term.replace('%', ' ').replace('_', ' ').strip()}%"
        
        try:
            # Core Supabase Query: Uses ILIKE for case-insensitive partial match
            response = self.supabase.table(self.table_name) \
                .select("public_url, prompt, hash") \
                .ilike("prompt", safe_search_term) \
                .limit(limit) \
                .execute()

            if not response.data:
                logger.info("Supabase Image Search: No image found for this query.")
                # Fallback data for demonstration if DB is empty
                if "head model" in search_term.lower():
                    logger.info("Simulating image results for head model query.")
                    return [
                        {'prompt': 'Scanned 3D Head Model Data Quality Issues', 'public_url': 'https://placehold.co/400x300/1e293b/ffffff?text=Head+Model+Error', 'hash': 'sim1', 'source': 'Supabase Sim'},
                        {'prompt': 'Anthropometry Head Measurement Guide', 'public_url': 'https://placehold.co/400x300/22c55e/ffffff?text=Anthropometry+Guide', 'hash': 'sim2', 'source': 'Supabase Sim'},
                    ]
                return []

            # Format the results for internal use
            image_results = [
                {'prompt': item['prompt'], 'public_url': item['public_url'], 'hash': item['hash'], 'source': 'Supabase Image Store'}
                for item in response.data
            ]
            logger.info(f"Supabase Image Search: Found {len(image_results)} image(s).")
            return image_results

        except Exception as e:
            logger.error(f"Error connecting to Supabase or executing query: {e}")
            return []


# --- PDF Processing Functions (OMITTED FOR BREVITY, ASSUMED VALID) ---
def clean_text(text):
    # Placeholder
    return text

def curate_pdf_to_text(pdf_path_str, output_dir):
    # Placeholder
    return None

def load_documents_for_indexing(files=None):
    # Placeholder: Creates dummy files for demonstration if not found
    CURATED_DIR = 'curated_data_single_book'
    Path(CURATED_DIR).mkdir(parents=True, exist_ok=True)
    
    placeholder_filepaths = [Path(CURATED_DIR) / "dummy_file1.txt"]
    
    for f in placeholder_filepaths:
        if not os.path.exists(f):
            f.write_text("This is a placeholder document for the chatbot's RAG system.")
            logger.warning(f"Dummy file '{f}' created.")
            
    logger.info(f"Loading {len(placeholder_filepaths)} documents for indexing...")
    
    # Use SimpleDirectoryReader to load the text files
    reader = SimpleDirectoryReader(input_files=[str(p) for p in placeholder_filepaths], required_exts=[".txt"])
    documents = reader.load_data()
    
    for doc in documents:
        doc.metadata['category'] = "BookContent"
        doc.metadata['filename'] = os.path.basename(doc.metadata.get("file_path") or "unknown")

    logger.info(f"Loaded {len(documents)} document segments in total.")
    return documents

# --- RAG Strategy Implementations (OMITTED FOR BREVITY, ASSUMED VALID) ---
async def run_planning_workflow(query: str, agent_instance: ReActAgent, trace: List[str]) -> str:
    return "Planning workflow result simulated."

async def run_multi_step_query_engine_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    return "Multi-step query engine result simulated.", [], []

async def run_multi_strategy_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> Dict[str, Any]:
    return {"response": "Multi-strategy synthesis simulated.", "links": [], "local_files": []}

# --- Model Context Protocol Processing Function (MODIFIED WITH IMAGE ROUTING) ---
async def process_model_context_query(
    query: str,
    context_memory: Dict[str, Any],
    tool_outputs: List[Dict],
    scratchpad: str,
    agent_instance: ReActAgent,
    rac_corrector_instance: 'RACCorrector',
    testing_mode: bool,
    suppress_threshold: float,
    flag_threshold: float,
    selected_rag_strategy: str,
    selected_retrieval_method: str,
    local_query_engine: Any,
    google_custom_search_instance: Any,
    tools_for_agent: List[FunctionTool],
    supabase_image_tool: Any, # New required parameter
) -> Dict[str, Any]:
    
    response_trace = [f"ModelContextQuery received: Query='{query}'"]

    # 1. CLASSIFY INTENT
    intent_result = classify_intent_llm(query, Settings.llm)
    intent = intent_result.intent
    intent_query = intent_result.query_term
    logger.info(f"Intent Classification: {intent.upper()} for term '{intent_query}'")

    if intent == "image":
        # 2. IMAGE RETRIEVAL AND EARLY RETURN
        logger.info("Executing Image Retrieval Workflow...")
        image_results = supabase_image_tool.retrieve_image_urls(intent_query, limit=3)
        image_links_for_frontend = image_results # Already structured correctly

        if image_results:
            final_answer = f"✅ I found the following {len(image_results)} visual result(s) for '{intent_query}'."
            sources_used = {
                'local_sources_count': len(image_links_for_frontend),
                'local_files': [f"Image Prompt: {img['prompt']} (Hash: {img['hash']})" for img in image_results],
                'web_sources_count': 0, 
                'web_links': [], 
                'used_local': True, 
                'used_web': False
            }
            confidence = 1.0
        else:
            final_answer = f"❌ No matching images found in the ComFit image database for '{intent_query}'. I am now proceeding to standard text RAG for context."
            sources_used = {
                'local_sources_count': 0, 'local_files': [], 'web_sources_count': 0, 
                'web_links': [], 'used_local': False, 'used_web': False
            }
            confidence = 0.0

        # If image was successfully found, return immediately.
        if image_results:
            return {
                "final_answer": final_answer,
                "trace": [f"Image Intent Detected and Handled: {intent_query}"],
                "confidence_score": confidence,
                "sources_used": sources_used,
                "image_links": image_links_for_frontend
            }
        
        # If image was NOT found, we fall through to the RAG path below, but set the query
        # to the user's original query (or intent_query) and append the failure message.
        logger.info("Image lookup failed. Falling through to standard RAG/RAC workflow.")
        # We don't overwrite the original query, but include the failure message in the final answer
        
    # --- RAG/RAC EXECUTION (Executed if intent is 'text' OR if 'image' failed) ---

    # Simulate RAG execution 
    original_response_text = f"This is a RAG-generated answer for '{query}' using the {selected_rag_strategy} strategy."
    unique_local_filenames = ["Anthropometry_Book.txt"]
    unique_web_sources = []
    average_conf = 0.85

    # Simulate RAC application
    rac_result = rac_corrector_instance.correct_response(original_response_text, apply_corrections=not testing_mode)
    final_answer_content = rac_result['corrected_response']
    
    # Simulate Confidence Cascade
    if average_conf < suppress_threshold:
        final_answer = f"❌ Response suppressed due to very low confidence ({average_conf:.2f})."
    elif average_conf < flag_threshold:
        final_answer = f"⚠️ Low confidence in response ({average_conf:.2f}). Please use with caution.\n\n" + final_answer_content
    else:
        final_answer = final_answer_content

    # If the initial image search failed (and we fell through), prepend the failure message
    if intent == "image" and not image_results:
        final_answer = f"{final_answer_content}\n\n{final_answer}"

    return {
        "final_answer": final_answer,
        "trace": response_trace,
        "confidence_score": average_conf,
        "sources_used": {
            "local_sources_count": len(unique_local_filenames), 
            "local_files": unique_local_filenames, 
            "web_sources_count": len(unique_web_sources), 
            "web_links": unique_web_sources,
            "used_local": len(unique_local_filenames) > 0,
            "used_web": len(unique_web_sources) > 0
        },
        "image_links": [] # KEY: Empty list for the RAG/Text path
    }

def main():
    """
    Main function to set up and run the enhanced hybrid chatbot.
    Handles CLI interaction, RAG strategy selection, and image retrieval.
    """
    # Check for dry-run mode
    testing_mode_enabled = "--dry-run" in sys.argv
    if testing_mode_enabled:
        logger.info("RAC Testing Mode (--dry-run) enabled.")
        sys.argv.remove("--dry-run")

    # Ensure curated data directory exists
    CURATED_DATA_SINGLE_BOOK_DIR = 'curated_data_single_book'
    os.makedirs(CURATED_DATA_SINGLE_BOOK_DIR, exist_ok=True)
    
    # Load and index local PDF documents
    documents = load_documents_for_indexing()
    
    # --- RAG/Vector Store Setup (SIMULATED) ---
    # NOTE: In a real environment, this simulation would be replaced by actual indexing.
    class MockQueryEngine:
        def query(self, query):
            # Simplified mock for demonstration
            return Response(response="Simulated data from local vector store.", source_nodes=[])
    
    local_query_engine = MockQueryEngine()
    logger.info("Local PDF data indexing simulated successfully.")

    # Validate Google API keys
    google_api_key, google_cse_id = validate_google_api_keys_from_env()
    if not (google_api_key and google_cse_id):
        logger.critical("FATAL ERROR: Google API keys not configured. Exiting.")
        sys.exit(1)
    
    # Initialize Google Custom Search Tool
    Google_Search_instance = GoogleCustomSearchTool(
        api_key=google_api_key,
        cse_id=google_cse_id,
        num_results=5 
    )
    
    # Initialize Supabase Image Retrieval Tool
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

    if not (SUPABASE_URL and SUPABASE_ANON_KEY):
        logger.critical("FATAL ERROR: Supabase keys not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file. Exiting.")
        sys.exit(1)
    
    supabase_image_tool = SupabaseImageTool(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_ANON_KEY
    )

    # Initialize RAC Corrector (using mocks for dependencies)
    rac_corrector = RACCorrector(
        llm=OPTIMAL_LLM,
        local_query_engine=local_query_engine,
        Google_Search_tool=Google_Search_instance
    )
    rac_corrector.testing_mode = testing_mode_enabled
    
    # --- Agent Setup (SIMULATED) ---
    # Create LlamaIndex FunctionTool mocks
    local_rag_tool = FunctionTool.from_defaults(fn=lambda q: "Local data available.", name="local_book_qa", description="Queries local PDF content.")
    Google_Search_tool_for_agent = FunctionTool.from_defaults(fn=Google_Search_instance.search_legacy, name="google_web_search", description="Queries the internet.")
    tools_for_agent = [local_rag_tool, Google_Search_tool_for_agent]

    class MockAgent:
        async def chat(self, query):
            return Response(response=f"Agent response for: {query}")
    agent = MockAgent()
    
    logger.info("Initialized Enhanced Hybrid Chatbot with Image Retrieval.")
    logger.info("\n--- Enhanced Hybrid Chatbot with Model Context Protocol READY ---")
    
    # --- Configuration and Loop Setup (OMITTED FOR BREVITY, ASSUMED VALID) ---
    SUPPRESS_THRESHOLD = 0.4
    FLAG_THRESHOLD = 0.6
    RAG_STRATEGIES = {"1": "rac_enhanced_hybrid_rag"}
    STRATEGY_NAMES = {"rac_enhanced_hybrid_rag": "RAC Enhanced Hybrid RAG"}
    RETRIEVAL_METHODS = {"hybrid": "Local and Web"}
    current_rag_strategy = "1"
    current_retrieval_method = "hybrid"
    context_memory = {}
    
    # Main conversational loop
    while True:
        # Simplified selection logic for demonstration
        user_question = input("Enter question (e.g., 'show me anthropometry' or 'what is RAG'): ").strip()
        
        if user_question.lower() == 'exit':
            break

        print("\n" + "="*50)
        print("--- Agent's Response (via Model Context Protocol) ---")
        print("="*50)

        start_time = time.time()
        
        # --- Pre-Step: Classify User Intent ---
        intent_result = classify_intent_llm(user_question, OPTIMAL_LLM)
        intent = intent_result.intent
        intent_query = intent_result.query_term
        logger.info(f"Detected Intent: {intent.upper()} with query term: '{intent_query}'")
        
        mcp_response = None
        
        # --- RAG/RAC BRANCH EXECUTION (The client is responsible for calling this) ---
        try:
            mcp_response = asyncio.run(process_model_context_query(
                query=user_question,
                context_memory=context_memory, 
                tool_outputs=[], 
                scratchpad="",   
                agent_instance=agent,
                rac_corrector_instance=rac_corrector,
                testing_mode=testing_mode_enabled,
                suppress_threshold=SUPPRESS_THRESHOLD,
                flag_threshold=FLAG_THRESHOLD,
                selected_rag_strategy=RAG_STRATEGIES[current_rag_strategy],
                selected_retrieval_method=current_retrieval_method,
                local_query_engine=local_query_engine,
                google_custom_search_instance=Google_Search_instance,
                tools_for_agent=tools_for_agent,
                supabase_image_tool=supabase_image_tool 
            ))
        except Exception as e:
            logger.error(f"Error during interaction loop: {e}", exc_info=True)
            print("An unhandled error occurred while processing your request. Please check logs for details.")
            continue

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
        
        # Display the final answer, image links, and sources
        print(mcp_response["final_answer"])
        if mcp_response.get("image_links"):
            print("\n🖼️ **RETRIEVED IMAGES:**")
            for link in mcp_response["image_links"]:
                print(f"  - Title: {link['prompt']}")
                print(f"    URL: {link['public_url']}")
        
        if 'sources_used' in mcp_response:
            sources_info_str = format_sources_info(mcp_response["sources_used"])
            print(sources_info_str)

        print("="*50 + "\n")
        
        # Store the current interaction in context memory
        context_memory[user_question] = mcp_response["final_answer"]
            
    logger.info("Enhanced Hybrid Chatbot with Model Context Protocol Session Completed.")

def format_sources_info(sources_info: Dict[str, Any]) -> str:
    """
    Formats the sources information into a user-friendly string for display.
    """
    if not sources_info:
        return "\n📚 **Sources Used:** None"
    
    info_lines = ["\n📚 **Sources Used:**"]
    
    local_files = sources_info.get('local_files', [])
    if sources_info.get('used_local', False) and local_files:
        info_lines.append(f"  📄 **Local Documents/Images Referenced:**")
        for i, filename in enumerate(local_files[:5], 1): # Limit to top 5 for brevity
            info_lines.append(f"    {i}. {filename}")
    
    if sources_info.get('used_web', False):
        info_lines.append(f"  🌐 Web Search: {sources_info['web_sources_count']} queries")
        
        web_links = sources_info.get('web_links', [])
        if web_links:
            info_lines.append(f"\n🔗 **Web Sources Referenced (Top 5):**")
            for i, link in enumerate(web_links[:5], 1):
                title = link.get('title', 'Unknown Title')
                url = link.get('url', '')
                info_lines.append(f"  {i}. **{title}**")
                info_lines.append(f"    URL: {url}")
            
    if not sources_info.get('used_local', False) and not sources_info.get('used_web', False):
        info_lines.append("  ℹ️ No external sources were consulted.")
    
    return "\n".join(info_lines)
