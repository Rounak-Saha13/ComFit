
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
from pydantic import BaseModel, Field

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

# --- Image Integration Global Setting (NEW ADDITION) ---
SUPABASE_URL = "https://tyswhhteurchuzkngqja.supabase.co/storage/v1/object/public/comfit_images/"


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

# --- RAC (Retrieval-Augmented Correction) Implementation ---
class FactualClaimExtractor:
    """
    Extracts atomic factual claims from a given text using an LLM.
    These claims are then used for factual verification.
    """
    def __init__(self, llm):
        self.llm = llm
    
    def extract_claims(self, text: str) -> List[str]:
        logger.info("Extracting factual claims from the response...")
        # Prompt to instruct the LLM to extract atomic factual claims
        extraction_prompt = f"""
        Task: Extract atomic factual claims from the following text. 
        An atomic factual claim is a single, verifiable statement that can be true or false.
        
        Rules:
        1. Each claim should be independent and verifiable
        2. Break down complex sentences into simple facts
        3. Include numerical facts, dates, names, and specific details
        4. Ignore opinions, subjective statements, or procedural instructions
        5. Do NOT extract claims about the model's internal process or tools
        6. Return each claim on a separate line starting with "CLAIM:"
        
        Text to analyze:
        {text}
        
        Extract the factual claims:
        """
        try:
            response = self.llm.complete(extraction_prompt)
            claims = []
            for line in str(response).split('\n'):
                line = line.strip()
                if line.startswith('CLAIM:'):
                    claim = line.replace('CLAIM:', '').strip()
                    if claim and len(claim) > 10: # Filter out very short or empty claims
                        claims.append(claim)
            logger.info(f"Extracted {len(claims)} factual claims")
            return claims
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

class FactVerifier:
    """
    Verifies factual claims against local documents and web search results.
    It supports different retrieval methods and caches verification results.
    """
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.local_query_engine = local_query_engine
        self.Google_Search_tool = Google_Search_tool
        self.retrieval_method = retrieval_method # 'local', 'web', 'hybrid', 'automatic'
        self.verification_cache = {} # Cache to store previously verified claims
        self.reversal_min_confidence = 0.95 # Minimum confidence required for a factual reversal

    def _get_claim_hash(self, claim: str) -> str:
        """Generates a hash for a claim for caching purposes."""
        return hashlib.md5(claim.lower().encode('utf-8')).hexdigest()

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verifies a single factual claim by searching local and/or web sources.
        Returns a dictionary indicating support, confidence, evidence, and correction suggestions.
        """
        claim_hash = self._get_claim_hash(claim)
        if claim_hash in self.verification_cache:
            logger.info(f"Cache hit for claim: {claim[:50]}...")
            return self.verification_cache[claim_hash]
        
        logger.info(f"Verifying claim: {claim[:50]}... with retrieval method: {self.retrieval_method}")
        evidence = []
        web_sources = []  # To track structured web source links
        local_source_files = [] # To track specific local filenames

        use_local_source, use_web_source = False, False

        # Logic to determine which sources to use based on retrieval method
        if self.retrieval_method == "local":
            use_local_source = True
        elif self.retrieval_method == "web":
            use_web_source = True
        elif self.retrieval_method == "hybrid":
            use_local_source = True
            use_web_source = True
        elif self.retrieval_method == "automatic":
            # Heuristic to prioritize local search for specific keywords
            local_keywords = ["speed process", "anthropometry", "product fit", "sizing", "book", "document", "pdf", "chapter", "section"]
            if any(term in claim.lower() for term in local_keywords):
                use_local_source = True
            use_web_source = True # Always use web as a fallback for automatic mode

        if use_local_source:
            try:
                logger.info("Checking claim against local PDF content...")
                queries_to_try = [
                    claim,
                    f"What information is available about: {claim}",
                    f"Find details related to: {self._extract_search_terms(claim)}",
                    f"Does the document mention: {self._extract_search_terms(claim)}"
                ]
                local_evidence_found = False
                for query in queries_to_try:
                    try:
                        # local_query_engine.query returns a Response object
                        local_result = self.local_query_engine.query(query) 
                        local_content = str(local_result).strip()
                        extracted_files = _extract_source_filenames(local_result)
                        
                        if (local_content and len(local_content) > 20 and
                                not any(phrase in local_content.lower() for phrase in [
                                        "i don't know", "no information", "not mentioned",
                                        "cannot find", "not available", "no details"])):
                            evidence.append({
                                'source': 'local_knowledge',
                                'content': local_content,
                                'confidence': 0.9, # Higher confidence for local, trusted data
                                'query_used': query,
                                'filenames': extracted_files # Store filenames here
                            })
                            local_source_files.extend(extracted_files)
                            local_evidence_found = True
                            logger.info(f"Local evidence found for query: {query[:50]}...")
                            break # Stop on first relevant local evidence
                    except Exception as e:
                        logger.warning(f"Local query failed for '{query[:30]}...': {e}")
                        continue
                if not local_evidence_found:
                    logger.info("No relevant local evidence found.")
            except Exception as e:
                logger.warning(f"Local verification failed: {e}")
        
        # Only use web if no local evidence found or if local is not chosen
        if use_web_source and not evidence: 
            try:
                logger.info(f"Searching web for: {claim[:50]}...")
                search_query = self._extract_search_terms(claim)
                # Call the Google Search tool and get both text and structured links
                web_result_text, web_links_from_search = self.Google_Search_tool.search(search_query) 
                if web_result_text and "No relevant search results" not in web_result_text:
                    evidence.append({
                        'source': 'web_search',
                        'content': web_result_text,
                        'confidence': 0.7, # Moderate confidence for web search
                        'query_used': search_query
                    })
                    web_sources.extend(web_links_from_search) # Store the structured web source links
                    logger.info(f"Web evidence found for query: {search_query}")
            except Exception as e:
                logger.warning(f"Web verification failed: {e}")
        
        evidence_sources = [e['source'] for e in evidence]
        logger.info(f"Evidence sources used: {evidence_sources}")
        verification_result = self._analyze_evidence(claim, evidence)
        
        # Store result in cache before returning
        result_to_cache = {
            'claim': claim,
            'is_supported': verification_result['is_supported'],
            'confidence': verification_result['confidence'],
            'evidence': evidence,
            'web_sources': web_sources, # Add web sources to result
            'local_source_files': sorted(list(set(local_source_files))), # Add unique local filenames
            'correction_suggestion': verification_result.get('correction', None),
            'warning': verification_result.get('warning', None)
        }
        self.verification_cache[claim_hash] = result_to_cache
        return result_to_cache
    
    def _extract_search_terms(self, claim: str) -> str:
        """Extracts key terms from a claim for more effective search queries."""
        words = claim.split()
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
        key_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
        return ' '.join(key_words[:5]) # Return up to 5 key words
    
    def _analyze_evidence(self, claim: str, evidence: List[Dict]) -> Dict[str, Any]:
        """
        Analyzes the collected evidence to determine if the claim is supported,
        contradicted, or if there's insufficient evidence.
        Uses an LLM for nuanced analysis and correction suggestions.
        """
        if not evidence:
            return {
                'is_supported': False,
                'confidence': 0.0,
                'correction': None,
                'warning': "No evidence found to verify this claim."
            }
        
        # Separate local and web evidence for prioritized analysis
        local_evidence = [e for e in evidence if e['source'] == 'local_knowledge']
        web_evidence = [e for e in evidence if e['source'] == 'web_search']
        
        evidence_text = ""
        if local_evidence:
            evidence_text += "=== LOCAL DOCUMENT EVIDENCE ===\n"
            for e in local_evidence:
                filenames_str = f" (Files: {', '.join(e['filenames'])})" if e.get('filenames') else ""
                evidence_text += f"Source: {e['source']}{filenames_str} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        if web_evidence:
            evidence_text += "=== WEB SEARCH EVIDENCE ===\n"
            for e in web_evidence:
                evidence_text += f"Source: {e['source']} (Query: {e.get('query_used', 'N/A')})\n{e['content']}\n\n"
        
        # LLM prompt to analyze the claim against the evidence
        analysis_prompt = f"""
        Task: Analyze whether the CLAIM is supported by the EVIDENCE.
        CLAIM: {claim}
        EVIDENCE:
        {evidence_text}
        Instructions:
        1. Prioritize LOCAL DOCUMENT EVIDENCE for specific terms or definitions.
        2. Determine if SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE.
        3. Provide confidence score (0.0 to 1.0).
        4. If contradicted, suggest a correction.
        5. For reversals (e.g., changing a positive statement to a negative one, or vice-versa),
           require confidence >= {self.reversal_min_confidence}.
        Respond:
        VERDICT: [SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE]
        CONFIDENCE: [0.0-1.0]
        CORRECTION: [If contradicted, provide correction, else "None"]
        REASONING: [Brief explanation]
        """
        
        try:
            response = str(self.llm.complete(analysis_prompt))
            # Parse LLM response to extract verdict, confidence, and correction
            verdict = "INSUFFICIENT_EVIDENCE"
            confidence = 0.5
            correction = None
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('VERDICT:'):
                    verdict = line.replace('VERDICT:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except ValueError:
                        confidence = 0.5 # Default if parsing fails
                elif line.startswith('CORRECTION:'):
                    correction_text = line.replace('CORRECTION:', '').strip()
                    if correction_text.lower() != 'none':
                        correction = correction_text
            
            # Boost confidence if local evidence was primary and relevant
            if local_evidence and verdict in ["SUPPORTED", "CONTRADICTED"]:
                confidence = min(confidence + 0.2, 1.0) # Add a small boost for local sources
            
            is_supported = verdict == "SUPPORTED"
            # Logic to detect potential factual reversals (positive to negative, or vice-versa)
            negative_markers = ["not", "no", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "never", "false", "untrue", "without"]
            original_has_neg = any(neg_m in claim.lower() for neg_m in negative_markers)
            correction_has_neg = any(neg_m in (correction or "").lower() for neg_m in negative_markers)
            
            # If a reversal is detected, check if confidence is high enough
            if correction and ((original_has_neg and not correction_has_neg) or (not original_has_neg and correction_has_neg)):
                if confidence < self.reversal_min_confidence:
                    logger.warning(f"Potential reversal detected but confidence ({confidence:.2f}) below threshold.")
                    correction = None # Suppress correction if confidence is too low
                    verdict = "INSUFFICIENT_EVIDENCE"
                    confidence = 0.5
            
            warning_message = None
            if not is_supported and confidence <= 0.6:
                # Flag responses with low confidence, even if no direct contradiction
                warning_message = f"Low confidence in claim: {claim}. Verdict: {verdict}. Confidence: {confidence:.2f}."
                logger.warning(warning_message)
            
            return {
                'is_supported': is_supported,
                'confidence': confidence,
                'correction': correction,
                'warning': warning_message
            }
        except Exception as e:
            logger.error(f"Error analyzing evidence: {e}")
            return {
                'is_supported': False,
                'confidence': 0.0,
                'correction': None,
                'warning': f"Error during evidence analysis: {e}"
            }

class RACCorrector:
    """
    Orchestrates the Retrieval-Augmented Correction (RAC) process.
    It extracts claims, verifies them, and applies corrections to the original response.
    """
    def __init__(self, llm, local_query_engine, Google_Search_tool, retrieval_method="hybrid"):
        self.llm = llm
        self.claim_extractor = FactualClaimExtractor(llm)
        self.fact_verifier = FactVerifier(llm, local_query_engine, Google_Search_tool, retrieval_method)
        self.retrieval_method = retrieval_method
        self.correction_threshold = 0.5 # Minimum confidence to apply a correction
        self.uncertainty_threshold = 0.6 # Threshold to flag claims as uncertain
        self.local_priority = True # Not directly used in current logic, but can be for future weighting
        self.testing_mode = False # If true, corrections are not applied, only reported
        self.rac_enabled = True # Master switch for RAC

    def correct_response(self, original_response: str, apply_corrections: bool = True) -> Dict[str, Any]:
        """
        Applies RAC to an original LLM response.
        If RAC is disabled or no claims are extracted, returns the original response.
        """
        if not self.rac_enabled:
            logger.info("RAC is disabled, skipping correction.")
            return {
                'original_response': original_response,
                'corrected_response': original_response,
                'claims_analyzed': 0,
                'corrections_made': 0,
                'verification_results': [],
                'uncertain_claims': [],
                'average_confidence': 1.0 # If RAC is off, assume full confidence
            }
        
        logger.info("Starting RAC correction process...")
        start_claim_extraction = time.perf_counter()
        claims = self.claim_extractor.extract_claims(original_response)
        end_claim_extraction = time.perf_counter()
        logger.info(f"Timing - Claim Extraction: {end_claim_extraction - start_claim_extraction:.4f} seconds")
        
        if not claims:
            logger.info("No factual claims extracted, returning original response")
            return {
                'original_response': original_response,
                'corrected_response': original_response,
                'claims_analyzed': 0,
                'corrections_made': 0,
                'verification_results': [],
                'uncertain_claims': [],
                'average_confidence': 0.0 # Cannot assess confidence if no claims
            }
            
        verification_results = []
        corrections_needed = []
        uncertain_claims = []
        
        # Ensure the FactVerifier instance uses the current retrieval mode
        self.fact_verifier.retrieval_method = self.retrieval_method
        
        start_verification = time.perf_counter()
        for i, claim in enumerate(claims, 1):
            logger.info(f"Processing claim {i}/{len(claims)}: {claim[:50]}...")
            start_single_verify = time.perf_counter()
            result = self.fact_verifier.verify_claim(claim)
            end_single_verify = time.perf_counter()
            logger.info(f"Timing - Single Claim Verification ({i}): {end_single_verify - start_single_verify:.4f} seconds")
            
            verification_results.append(result)
            evidence_sources = [e['source'] for e in result['evidence']]
            logger.info(f"Claim {i} verification: {result['is_supported']}, confidence: {result['confidence']:.2f}, sources: {evidence_sources}")
            
            # If claim is not supported and confidence is above threshold, add to corrections
            if not result['is_supported'] and result['confidence'] > self.correction_threshold:
                if result['correction_suggestion']:
                    corrections_needed.append({
                        'original_claim': claim,
                        'correction': result['correction_suggestion'],
                        'confidence': result['confidence'],
                        'evidence_sources': evidence_sources,
                        'local_source_files': result.get('local_source_files', [])
                    })
                    logger.info(f"Correction needed for claim {i}")
                else:
                    logger.info(f"Claim {i} not supported but no correction available")
            
            # If confidence is below uncertainty threshold, flag it
            if result['confidence'] < self.uncertainty_threshold:
                uncertain_claims.append({
                    'claim': claim,
                    'confidence': result['confidence'],
                    'verdict': "SUPPORTED" if result['is_supported'] else "CONTRADICTED" if result['correction_suggestion'] else "INSUFFICIENT_EVIDENCE",
                    'warning': result.get('warning', 'Low confidence.')
                })
                logger.warning(f"Claim {i} is uncertain: {claim[:50]}... Confidence: {result['confidence']:.2f}")
        
        end_verification = time.perf_counter()
        logger.info(f"Timing - All Claims Verification: {end_verification - start_verification:.4f} seconds")
        
        corrected_response = original_response
        # Apply corrections only if not in testing mode and corrections are needed
        if apply_corrections and not self.testing_mode and corrections_needed:
            start_apply_corrections = time.perf_counter()
            corrected_response = self._apply_corrections(original_response, corrections_needed)
            end_apply_corrections = time.perf_counter()
            logger.info(f"Timing - Applying Corrections: {end_apply_corrections - start_apply_corrections:.4f} seconds")
            
        logger.info(f"RAC correction completed. Analyzed {len(claims)} claims, made {len(corrections_needed)} corrections")
        
        total_confidence = sum(res['confidence'] for res in verification_results)
        average_confidence = total_confidence / len(claims) if claims else 0.0 # Calculate average confidence
        
        return {
            'original_response': original_response,
            'corrected_response': corrected_response,
            'claims_analyzed': len(claims),
            'corrections_made': len(corrections_needed),
            'verification_results': verification_results,
            'corrections_applied': corrections_needed,
            'uncertain_claims': uncertain_claims,
            'average_confidence': average_confidence
        }
    
    def _apply_corrections(self, original_response: str, corrections: List[Dict]) -> str:
        """
        Uses an LLM to integrate the suggested corrections back into the original response,
        maintaining natural language flow.
        """
        correction_prompt = f"""
        Task: Apply corrections to the original response while maintaining its structure and flow.
        ORIGINAL RESPONSE:
        {original_response}
        CORRECTIONS TO APPLY:
        {chr(10).join([f"- Replace/correct: '{c['original_claim']}' -> '{c['correction']}'" for c in corrections])}
        Instructions:
        1. Integrate corrections naturally
        2. Maintain original tone and structure
        3. Ensure response flows well
        4. Don't add unnecessary information
        5. Keep response length similar
        6. Do NOT mention correction process
        Provide the corrected response:
        """
        try:
            corrected = str(self.llm.complete(correction_prompt))
            return corrected.strip()
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            return original_response # Return original if correction application fails

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
    """
    A custom tool to perform Google Custom Search requests.
    It returns both formatted text results and structured links.
    """
    def __init__(self, api_key: str, cse_id: str, num_results: int = 3):
        self.api_key = api_key
        self.cse_id = cse_id
        self.num_results = num_results
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str) -> tuple[str, list]:
        """
        Performs a Google Custom Search for the given query.
        Returns a formatted string of results and a list of structured link dictionaries.
        """
        logger.info(f"Google Search: '{query}'")
        start_web_api = time.perf_counter()
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": self.num_results
        }
        links = []  # To store structured link information
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            search_results = response.json()
            formatted_results = []
            if "items" in search_results:
                for i, item in enumerate(search_results["items"]):
                    title = item.get('title', 'N/A')
                    snippet = item.get('snippet', 'N/A')
                    link = item.get('link', 'N/A')
                    
                    # Store structured link info
                    links.append({
                        'title': title,
                        'url': link,
                        'snippet': snippet[:100] + "..." if len(snippet) > 100 else snippet
                    })
                    
                    # Format results for the LLM
                    formatted_results.append(
                        f"Result {i+1}: Title: {title}\n"
                        f"Snippet: {snippet}\n"
                        f"Link: {link}\n"
                        f"---"
                    )
                return "\n".join(formatted_results), links
            else:
                return "No relevant search results found.", []
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Search API error: {e}")
            return f"Error performing web search: {str(e)}", []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return "Error processing web search results.", []
        finally:
            end_web_api = time.perf_counter()
            logger.info(f"Timing - Google Web Search API Call: {end_web_api - start_web_api:.4f} seconds")

    def search_legacy(self, query: str) -> str:
        """
        Legacy method for backward compatibility, returning only the formatted text.
        This is used when the LlamaIndex agent expects a single string return from a tool.
        """
        result, _ = self.search(query)
        return result

# --- PDF Processing Functions ---
def clean_text(text):
    """
    Cleans extracted text from PDFs by removing hyphenation,
    standardizing newlines, removing extra spaces, and page markers.
    """
    # Remove hyphenation at line breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace newlines after punctuation with a space
    text = re.sub(r'[.!?]\n', '. ', text)
    text = re.sub(r'[,;]\n', ', ', text)
    # Replace all remaining newlines with spaces
    text = text.replace('\n', ' ')
    # Replace multiple spaces with a single space and strip
    text = re.sub(r'\s{2,}', ' ', text).strip()
    # Remove PDF page markers if present
    text = re.sub(r'--- PAGE \d+ ---', '', text)
    # Remove isolated page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text

def curate_pdf_to_text(pdf_path_str, output_dir):
    """
    Extracts text from a PDF, cleans it, and saves it to a text file.
    Uses a temporary file for processing to avoid issues with original paths.
    """
    pdf_path = Path(pdf_path_str)
    
    if not pdf_path.is_file():
        logger.critical(f"FATAL ERROR: PDF file not found at '{pdf_path_str}'. Exiting.")
        sys.exit(1)

    # Create a temporary directory for safe PDF copying and processing
    temp_dir = tempfile.mkdtemp()
    sanitized_filename = "temp_pdf.pdf"
    temp_pdf_path = Path(temp_dir) / sanitized_filename

    try:
        # Copy the original PDF to the temporary location
        shutil.copy(pdf_path, temp_pdf_path)
        logger.info(f"Copied original PDF to temporary path: {temp_pdf_path}")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not copy PDF file from '{pdf_path_str}' to temporary location. Error: {e}. Exiting.")
        shutil.rmtree(temp_dir) # Clean up temp directory on error
        sys.exit(1)

    txt_filename = pdf_path.stem + '.txt'
    output_filepath = Path(output_dir) / txt_filename
    
    logger.info(f"Processing PDF: {pdf_path.name}...")
    full_text_pages = []
    try:
        with open(temp_pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    full_text_pages.append(text)
            combined_text = "\n\n".join(full_text_pages)
            final_curated_text = clean_text(combined_text)
            if not final_curated_text.strip():
                logger.warning(f"Extracted text from '{pdf_path}' is empty. Skipping.")
                shutil.rmtree(temp_dir)
                return None
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                outfile.write(final_curated_text)
            logger.info(f"Curated and saved text to: {output_filepath}")
            return str(output_filepath)
    except PyPDF2.errors.PdfReadError:
        logger.critical(f"FATAL ERROR: Could not read PDF '{temp_pdf_path}'. Ensure it's a valid PDF. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"FATAL ERROR: Error processing '{temp_pdf_path}': {e}. Exiting.")
        sys.exit(1)
    finally:
        shutil.rmtree(temp_dir) # Always clean up temporary directory

# Corrected filepaths list (ensure no invisible characters or missing commas)
# These paths are specific to the user's local system and should be adjusted if moved.
filepaths = [
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Statistics & Experiment Design\_Multivariate Data Analysis_Hair.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\3DAnthropometryAndApplications.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\3DAnthropometryWearableProductDesign.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\3DLaserScanner.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\10.1201_9781003006091_previewpdf.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\9781439808801 (1).txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Materials & Manufacturing\biofunctional-textiles 8.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\Bodyspace Anthropometry, Ergonomics and the Design of the Work, Second Edition 1.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\DHM-HCII2019Book2.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Statistics & Experiment Design\Douglas-C.-Montgomery-Design-and-Analysis-of-Experiments-Wiley-2012.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Ergonomics\Ergonomic_Office_Workstation_Design_that_Conforms_.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Body Modeling\GARIProceedings.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Standards & References\Human Dimension and Interior Space A Source Book of Design Reference Standards2.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Statistics & Experiment Design\StatisticalModelHumanShapeAndPose.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Sustainability\865129978-Sustainable-Product-Design-And-Development-Anoop-Desai-Anil-Mital-pdf-download.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Fit & Sizing\9780429327803_googlepreview.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Fit & Sizing\ATOB-V12_Iss1_Article24.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Sustainability\A_Study_of_Sustainable_Product_Design_Evaluation_B.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Wearable Design\Design-of-head-mounteddisplays-Zhang.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Wearable Design\sensors-24-04616.txt",
    r"D:\Sahithi\Sahithi3090\comfit main\categorized_data\Wearable Design\WEARABLE-TECHNOLOGIES3.txt",
]


def load_documents_for_indexing(files=None):
    """
    Loads and tags documents from a list of specified file paths into LlamaIndex Document objects.
    Performs a critical check for file existence.
    """
    if files is None:
        files = filepaths  # default to predefined list from global scope

    # Sanity check: ensure all specified files exist
    for f in files:
        if not os.path.exists(f):
            logger.critical(f"FATAL ERROR: Text file '{f}' not found. Exiting.")
            sys.exit(1)

    logger.info(f"Loading {len(files)} documents for indexing...")

    # Use SimpleDirectoryReader to load the text files
    reader = SimpleDirectoryReader(input_files=files, required_exts=[".txt"])
    documents = reader.load_data()

    if not documents:
        logger.critical("FATAL ERROR: No content loaded from provided files. Ensure files are not empty. Exiting.")
        sys.exit(1)

    # Attach metadata to each document for better retrieval and organization
    for doc in documents:
        doc_path = doc.metadata.get("file_path") or "unknown"
        doc.metadata['category'] = "BookContent" # General category for local documents
        doc.metadata['filename'] = os.path.basename(doc_path) # Original filename

    logger.info(f"Loaded {len(documents)} document segments in total.")
    return documents

# --- RAG Strategy Implementations ---
async def run_planning_workflow(query: str, agent_instance: ReActAgent, trace: List[str]) -> str:
    """
    Executes a query using the ReActAgent (planning workflow).
    The agent uses its tools to plan and execute steps to answer the query.
    """
    trace.append(f"Strategy: Planning Workflow - Agent thinking on '{query}'...")
    try:
        # Agent.chat() returns an AgentChatResponse which has a 'response' attribute
        # Note: The agent's response text will include the JSON string from local_book_qa_function
        response_obj = await asyncio.to_thread(agent_instance.chat, query)
        response = response_obj.response
        trace.append(f"Planning Workflow Raw Response: {response}")
        return response
    except Exception as e:
        trace.append(f"Error in Planning Workflow: {e}")
        logger.error(f"Error running planning workflow: {e}", exc_info=True)
        return "An error occurred during the planning workflow."

async def run_multi_step_query_engine_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> Tuple[str, List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """
    Executes a query using a RouterQueryEngine, which selects between local and web query engines.
    Returns the response text, any web links collected, local source filenames, and image links.
    """
    trace.append(f"Strategy: Multi-Step Query Engine - Routing '{query}'...")
    
    # Custom Google Query Engine that synthesizes results and returns links
    class GoogleQueryEngine:
        def __init__(self, search_tool_instance: GoogleCustomSearchTool, llm: Ollama):
            self.search_tool = search_tool_instance
            self.llm = llm
        
        async def aquery(self, query_str: str) -> Response:
            """Asynchronously queries Google Search and synthesizes an answer."""
            raw_search_result_text, links = await asyncio.to_thread(self.search_tool.search, query_str)
            
            if "No relevant search results" in raw_search_result_text:
                synthesized_answer = "No relevant information found on the web."
                return Response(response=synthesized_answer, metadata={"source": "Google Search", "links": links})

            synthesis_prompt = f"""
            Based on the following web search results, provide a concise and direct answer to the question: "{query_str}".
            Web Search Results:
            {raw_search_result_text}
            If the results do not contain a clear answer, state that.
            Provide only the answer, without referring to the search process or tools used.
            """
            try:
                synthesized_answer = await asyncio.to_thread(self.llm.complete, synthesis_prompt)
                synthesized_answer = str(synthesized_answer)
            except Exception as e:
                logger.error(f"Error during Google search synthesis: {e}")
                synthesized_answer = "Could not synthesize an answer from web search results."
            # Return Response object including metadata for links
            return Response(response=synthesized_answer, metadata={"source": "Google Search + LLM Synthesis", "links": links})

        def query(self, query_str: str) -> Response:
            """Synchronous wrapper for aquery for compatibility."""
            return asyncio.run(self.aquery(query_str))

    google_qe_instance = GoogleQueryEngine(google_custom_search_instance, OPTIMAL_LLM)
    
    query_engine_tools = []
    local_tool_used = False

    # Check for local tool presence (it will be of type FunctionTool if passed in tools_for_agent)
    local_tool_in_agent = next((tool for tool in tools_for_agent if tool.metadata.name == "local_book_qa"), None)

    if local_tool_in_agent:
        local_tool_used = True
        # NOTE: For RouterQueryEngine, we must use the raw QueryEngine (not the FunctionTool)
        local_tool_choice = QueryEngineTool.from_defaults(
            query_engine=local_query_engine,
            description=(
                "Useful for questions specifically about the content of the provided PDF book. "
                "Use when the question relates to 'speed process', 'anthropometry', 'product fit', 'sizing', etc."
            ),
        )
        query_engine_tools.append(local_tool_choice)

    if any(tool.metadata.name == "google_web_search" for tool in tools_for_agent):
        query_engine_tools.append(
            QueryEngineTool.from_defaults(
                query_engine=google_qe_instance,
                description="Useful for general knowledge questions, current events, or anything requiring internet search."
            )
        )

    if not query_engine_tools:
        return "No relevant query engine available for the selected retrieval method.", [], [], []

    # Initialize RouterQueryEngine to select the best tool
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=query_engine_tools,
        llm=OPTIMAL_LLM
    )
    
    try:
        response_obj = await asyncio.to_thread(router_query_engine.query, query)
        response_text = str(response_obj)
        web_links_collected = response_obj.metadata.get("links", []) # Extract links from metadata for web search
        local_files_collected = _extract_source_filenames(response_obj) # Extract filenames for local search
        
        # --- NEW IMAGE LOGIC FOR Multi-Step Query Engine ---
        image_links_collected = []
        if local_tool_used: # Check if the local query engine was potentially selected
            # Since the router selects a QueryEngine which returns a Response, we can directly inspect the nodes
            # Note: response_obj.source_nodes will only be populated if the Router selected the local_tool_choice.
            nodes = response_obj.source_nodes or []
            for node in nodes:
                name = node.metadata.get("name")
                caption = node.metadata.get("caption")
                page = node.metadata.get("page")
                if name:
                    image_links_collected.append({
                        "url": SUPABASE_URL + name,
                        "caption": caption,
                        "page": page,
                        "source_file": node.metadata.get('filename') # Add source filename for traceability
                    })

        trace.append(f"Multi-Step Query Engine Raw Response: {response_text}")
        return response_text, web_links_collected, local_files_collected, image_links_collected
    except Exception as e:
        trace.append(f"Error in Multi-Step Query Engine Workflow: {e}")
        logger.error(f"Error running multi_step_query_engine_workflow: {e}", exc_info=True)
        return "An error occurred during the multi-step query engine workflow.", [], [], []

async def run_multi_strategy_workflow(query: str, local_query_engine: Any, google_custom_search_instance: Any, trace: List[str], tools_for_agent: List[FunctionTool]) -> Dict[str, Any]:
    """
    Executes both local RAG and web search queries, then synthesizes a combined answer.
    Returns the synthesized response and all collected web links, local files, and images.
    """
    trace.append(f"Strategy: Multi-Strategy Workflow - Executing multiple queries for '{query}'...")
    responses = []
    all_links_from_web_search = [] # To collect links specifically from web searches
    all_local_files = [] # To collect local filenames from local RAG
    all_image_links = [] # To collect image links
    
    # Determine which sources are enabled by the provided tools
    use_local_source = any(tool.metadata.name == "local_book_qa" for tool in tools_for_agent)
    use_web_source = any(tool.metadata.name == "google_web_search" for tool in tools_for_agent)
    
    if use_local_source:
        try:
            # Use the local_book_qa_function to get the structured result
            local_response_dict_str = local_book_qa_function(query) 
            local_response_dict = json.loads(local_response_dict_str)

            local_response_text = local_response_dict['text']
            extracted_files = local_response_dict['local_files']
            images = local_response_dict['images']

            responses.append(f"Local RAG result: {local_response_text}")
            all_local_files.extend(extracted_files)
            all_image_links.extend(images) # Collect images
            
            trace.append(f"Multi-Strategy: Local RAG executed. Response snippet: {local_response_text[:100]}...")
            if extracted_files:
                trace.append(f"Multi-Strategy: Local RAG sources: {', '.join(extracted_files)}")
        except Exception as e:
            responses.append(f"Local RAG error: {e}")
            trace.append(f"Multi-Strategy: Local RAG error: {e}")
            logger.warning(f"Error in Multi-Strategy local RAG: {e}")
    
    if use_web_source:
        # Re-using the GoogleQueryEngine logic for consistency in link extraction
        class GoogleQueryEngineForMultiStrategy:
            def __init__(self, search_tool_instance: GoogleCustomSearchTool, llm: Ollama):
                self.search_tool = search_tool_instance
                self.llm = llm
            async def aquery(self, query_str: str) -> Response:
                raw_search_result_text, links = await asyncio.to_thread(self.search_tool.search, query_str)
                if "No relevant search results" in raw_search_result_text:
                    synthesized_answer = "No relevant information found on the web."
                    return Response(response=synthesized_answer, metadata={"source": "Google Search", "links": links})
                synthesis_prompt = f"""
                Based on the following web search results, provide a concise and direct answer to the question: "{query_str}".
                Web Search Results:
                {raw_search_result_text}
                If the results do not contain a clear answer, state that.
                Provide only the answer, without referring to the search process or tools used.
                """
                try:
                    synthesized_answer = await asyncio.to_thread(self.llm.complete, synthesis_prompt)
                    synthesized_answer = str(synthesized_answer)
                except Exception as e:
                    logger.error(f"Error during Google search synthesis (MultiStrategy): {e}")
                    synthesized_answer = "Could not synthesize an answer from web search results."
                return Response(response=synthesized_answer, metadata={"source": "Google Search + LLM Synthesis", "links": links})
            def query(self, query_str: str) -> Response:
                return asyncio.run(self.aquery(query_str))
        
        google_qe_instance = GoogleQueryEngineForMultiStrategy(google_custom_search_instance, OPTIMAL_LLM)
        try:
            web_response_obj = await asyncio.to_thread(google_qe_instance.query, query)
            web_response_text = str(web_response_obj)
            responses.append(f"Web Search result: {web_response_text}")
            trace.append(f"Multi-Strategy: Web Search executed. Response snippet: {web_response_text[:100]}...")
            all_links_from_web_search.extend(web_response_obj.metadata.get("links", [])) # Collect links
        except Exception as e:
            responses.append(f"Web Search error: {e}")
            trace.append(f"Multi-Strategy: Web Search error: {e}")
            logger.warning(f"Error in Multi-Strategy web search: {e}")
    
    combined_info = "\n\n".join(responses)
    if not combined_info.strip():
        return {"response": "No information found from any strategy.", "links": [], "local_files": [], "images": []}
    
    # Synthesize a final answer from all collected information
    synthesis_prompt = f"""
    Based on the following information from various sources, provide a comprehensive answer to the question: "{query}".
    Information:
    {combined_info}
    Instructions:
    - Synthesize the information coherently.
    - If conflicting details exist, reconcile based on source authority.
    - Do not mention sources by name.
    - If no relevant information is available, state that.
    """
    try:
        final_answer = await asyncio.to_thread(OPTIMAL_LLM.complete, synthesis_prompt)
        final_answer = str(final_answer)
        trace.append(f"Multi-Strategy Synthesis Complete. Final Answer snippet: {final_answer[:100]}...")
        return {"response": final_answer, "links": all_links_from_web_search, "local_files": all_local_files, "images": all_image_links}
    except Exception as e:
        trace.append(f"Error in Multi-Strategy Synthesis: {e}")
        logger.error(f"Error in multi-strategy synthesis: {e}", exc_info=True)
        return {"response": "An error occurred during multi-strategy synthesis.", "links": [], "local_files": [], "images": []}

# --- Model Context Protocol Processing Function ---
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
    tools_for_agent: List[FunctionTool]
) -> Dict[str, Any]:
    """
    Orchestrates the entire response generation pipeline, including:
    - Preprocessing the query
    - Executing the selected RAG strategy (Planning, Multi-Step, Multi-Strategy, No Method)
    - Applying Retrieval-Augmented Correction (RAC)
    - Handling confidence-based suppression/flagging
    - Collecting and formatting source information
    """
    logger.info(f"Processing Model Context Query: '{query}' with strategy: {selected_rag_strategy}, retrieval: {selected_retrieval_method}")
    response_trace = [f"ModelContextQuery received: Query='{query}'"]
    response_trace.append(f"Selected RAG Strategy: {selected_rag_strategy}")
    response_trace.append(f"Selected Retrieval Method: {selected_retrieval_method}")

    # Initialize source tracking dictionaries
    sources_used = {
        'local_sources': [], # Tracks detailed info about local queries
        'web_sources': [],   # Tracks queries made to web search
        'web_links_used': [], # Stores actual URL details from web searches
        'local_files_used': [], # Stores unique filenames from local documents
        'image_links': [] # Stores image metadata
    }

    start_total_process_mcp = time.perf_counter()
    try:
        # --- Preprocessing ---
        start_preprocess = time.perf_counter()
        processed_question = query # Currently, simple pass-through
        response_trace.append(f"Pre-processed query: '{processed_question}'")
        end_preprocess = time.perf_counter()
        response_trace.append(f"Timing - Preprocessing: {end_preprocess - start_preprocess:.4f} seconds")

        # --- RAG Strategy Execution ---
        start_rag_strategy = time.perf_counter()
        original_response_text = ""
        # These are reset per query as they represent the context for *this* specific interaction
        tool_outputs = []  
        scratchpad = ""
        
        # Enhanced tool wrappers to capture source usage and parse complex tool output (text + images/files)
        # Note: We need a wrapper only if the RAG strategy is 'planning_workflow' or 'no_method'
        # For 'multi_step' and 'multi_strategy', the external function handles the complexity.

        class SourceTrackingLocalBookQA:
            def __init__(self, fn_to_wrap):
                self.fn_to_wrap = fn_to_wrap
            
            # This wrapper handles the new complex JSON output structure from local_book_qa_function
            def __call__(self, query: str) -> str:
                # The wrapped function returns a JSON string: {"text": "...", "local_files": [...], "images": [...]}
                result_json_str = self.fn_to_wrap(query) 
                
                try:
                    result_dict = json.loads(result_json_str)
                    
                    local_response_text = result_dict.get('text', 'No text response.')
                    extracted_files = result_dict.get('local_files', [])
                    images = result_dict.get('images', [])

                    # Log and track sources for external analysis
                    sources_used['local_sources'].append({
                        'query': query,
                        'source_type': 'PDF Documents',
                        'timestamp': time.time(),
                        'filenames': extracted_files
                    })
                    sources_used['local_files_used'].extend(extracted_files)
                    sources_used['image_links'].extend(images) # Collect image links here

                    # Format the response for the Agent's context (keep it simple text + files for LLM's reasoning)
                    formatted_agent_response = local_response_text
                    if extracted_files:
                        formatted_agent_response += f"\n\nLocal Sources: {', '.join(extracted_files)}"
                    
                    # NOTE: We DO NOT expose the full JSON or image URLs to the Agent's scratchpad, 
                    # as that is for the final output. The Agent only sees the text.
                    return formatted_agent_response
                except Exception as e:
                    logger.error(f"Error parsing local_book_qa function output: {e}", exc_info=True)
                    return f"Error processing local RAG result for query: {query}"


        class SourceTrackingWebSearch:
            def __init__(self, search_tool):
                self.search_tool = search_tool
            
            def __call__(self, query: str) -> str:
                logger.info(f"Web Search: Querying for '{query}'")
                result_text, links = self.search_tool.search(query) # Call the actual search tool
                
                sources_used['web_sources'].append({
                    'query': query,
                    'source_type': 'Web Search',
                    'timestamp': time.time()
                })
                sources_used['web_links_used'].extend(links) # Collect structured links
                
                return result_text # Agent expects a string


        # Create dynamically wrapped tools based on which tools are active for this query
        enhanced_tools = []
        for tool in tools_for_agent:
            if tool.metadata.name == "local_book_qa":
                # Need to pass the original local_book_qa_function which returns JSON string
                wrapped_fn = SourceTrackingLocalBookQA(local_book_qa_function)
                enhanced_tool = FunctionTool.from_defaults(
                    fn=wrapped_fn, # Wrap the original function
                    name="local_book_qa",
                    description=tool.metadata.description
                )
                enhanced_tools.append(enhanced_tool)
            elif tool.metadata.name == "google_web_search":
                enhanced_web_search = SourceTrackingWebSearch(google_custom_search_instance)
                enhanced_tool = FunctionTool.from_defaults(
                    fn=enhanced_web_search,
                    name="google_web_search",
                    description=tool.metadata.description
                )
                enhanced_tools.append(enhanced_tool)
        
        # Initialize an agent instance for *this specific query* with the appropriate tools
        agent_instance_for_query = ReActAgent.from_tools(
            llm=OPTIMAL_LLM,
            tools=enhanced_tools, # Provide the dynamically created and wrapped tools
            verbose=False,
            max_iterations=30
        )

        response_data = {} # Common structure for all strategies
        
        # Execute the selected RAG strategy
        if selected_rag_strategy in ["planning_workflow", "rac_enhanced_hybrid_rag", "no_method"]:
            # These strategies use the ReActAgent.chat which returns a text string.
            agent_response_obj = await asyncio.to_thread(agent_instance_for_query.chat, processed_question)
            original_response_text = agent_response_obj.response
            response_data = {
                'response': original_response_text,
                'links': sources_used['web_links_used'], # Collected by wrapper
                'local_files': sources_used['local_files_used'], # Collected by wrapper
                'images': sources_used['image_links'] # Collected by wrapper
            }
            tool_outputs.append({"tool": selected_rag_strategy, "result": original_response_text})
            response_trace.append(f"Agent raw response: '{original_response_text}'")

        elif selected_rag_strategy == "multi_step_query_engine":
            original_response_text, links_from_msqe, files_from_msqe, images_from_msqe = await run_multi_step_query_engine_workflow(
                processed_question, local_query_engine, google_custom_search_instance, response_trace, enhanced_tools
            )
            response_data = {
                'response': original_response_text,
                'links': links_from_msqe,
                'local_files': files_from_msqe,
                'images': images_from_msqe
            }
            tool_outputs.append({"tool": "multi_step_query_engine", "result": original_response_text})
            # Add sources collected by multi-step query engine
            sources_used['web_links_used'].extend(links_from_msqe)
            sources_used['local_files_used'].extend(files_from_msqe)
            sources_used['image_links'].extend(images_from_msqe)
        
        elif selected_rag_strategy == "multi_strategy_workflow":
            response_data = await run_multi_strategy_workflow(
                processed_question, local_query_engine, google_custom_search_instance, response_trace, enhanced_tools
            )
            original_response_text = response_data['response']
            tool_outputs.append({"tool": "multi_strategy_workflow", "result": original_response_text})
            # Links, files, and images are explicitly returned by run_multi_strategy_workflow
            sources_used['web_links_used'].extend(response_data['links'])
            sources_used['local_files_used'].extend(response_data['local_files'])
            sources_used['image_links'].extend(response_data['images'])

        else:
            original_response_text = "Invalid RAG strategy selected."
            response_data = {'response': original_response_text, 'links': [], 'local_files': [], 'images': []}
            logger.error(original_response_text)

        final_response_data = response_data # Store the full response data structure
        original_response_text = final_response_data['response'] # Extract the text response

        end_rag_strategy = time.perf_counter()
        response_trace.append(f"Timing - RAG Strategy ({selected_rag_strategy}) Execution: {end_rag_strategy - start_rag_strategy:.4f} seconds")

        # --- RAC Application ---
        final_answer_content = original_response_text
        average_conf = 1.0 # Default confidence if RAC is disabled or no claims are found
        rac_web_sources = [] # To collect web sources specifically from RAC's verification step
        rac_local_files = [] # To collect local source files specifically from RAC's verification step

        if rac_corrector_instance.rac_enabled:
            start_rac = time.perf_counter()
            response_trace.append("Applying RAC (Retrieval-Augmented Correction)...")
            
            # Crucial: Set the retrieval method on the RAC corrector instance before running,
            # so RAC uses the same source preference as the main RAG strategy.
            rac_corrector_instance.retrieval_method = selected_retrieval_method
            
            rac_result = rac_corrector_instance.correct_response(original_response_text, apply_corrections=not testing_mode)
            
            # Collect web sources and local files that RAC used for verification
            for verification in rac_result.get('verification_results', []):
                if 'web_sources' in verification:
                    rac_web_sources.extend(verification['web_sources'])
                if 'local_source_files' in verification:
                    rac_local_files.extend(verification['local_source_files'])
            
            end_rac = time.perf_counter()
            response_trace.append(f"Timing - RAC Process: {end_rac - start_rac:.4f} seconds")
            response_trace.append(f"RAC Analysis: {rac_result['claims_analyzed']} claims checked.")
            
            if rac_result['corrections_made'] > 0:
                response_trace.append(f"  Corrections Applied: {rac_result['corrections_made']}")
                for corr in rac_result['corrections_applied']:
                    response_trace.append(f"    - Original: '{corr['original_claim'][:70]}...' -> Corrected: '{corr['correction'][:70]}...'")
            
            if rac_result['uncertain_claims']:
                response_trace.append(f"  Uncertain Claims Flagged: {len(rac_result['uncertain_claims'])}")
                for uc in rac_result['uncertain_claims']:
                    response_trace.append(f"    - Claim: '{uc['claim'][:70]}...' (Conf: {uc['confidence']:.2f})")
            
            average_conf = rac_result['average_confidence']
            final_answer_content = rac_result['corrected_response'] if rac_result['corrections_made'] > 0 else original_response_text
            tool_outputs.append({"tool": "rac_corrector", "result": final_answer_content})
            
            # --- Confidence Cascade (Suppression/Flagging) ---
            if average_conf < suppress_threshold:
                final_answer_content = f"❌ Response suppressed due to very low confidence ({average_conf:.2f})."
                response_trace.append(f"Confidence Cascade: Response Suppressed (Avg Confidence: {average_conf:.2f})")
            elif average_conf < flag_threshold:
                final_answer_content = f"⚠️ Low confidence in response ({average_conf:.2f}). Please use with caution.\n\n" + final_answer_content
                response_trace.append(f"Confidence Cascade: Response Flagged (Avg Confidence: {average_conf:.2f})")
            else:
                response_trace.append(f"Confidence Cascade: Response Accepted (Avg Confidence: {average_conf:.2f})")

        # Combine all unique web sources and local files collected from direct tool calls and RAC verification
        all_web_sources = sources_used['web_links_used'] + rac_web_sources
        unique_web_sources = []
        seen_urls = set()
        for source in all_web_sources:
            if isinstance(source, dict) and 'url' in source:
                if source['url'] and source['url'] != 'N/A' and source['url'] not in seen_urls:
                    unique_web_sources.append(source)
                    seen_urls.add(source['url'])
        
        all_local_filenames = sources_used['local_files_used'] + rac_local_files
        unique_local_filenames = sorted(list(set(all_local_filenames))) # Ensure unique and sorted
        
        # Combine image links from all strategies (RAC does not generate new images)
        unique_image_links = []
        seen_image_urls = set()
        for img in sources_used['image_links']:
            if img['url'] not in seen_image_urls:
                unique_image_links.append(img)
                seen_image_urls.add(img['url'])


        final_answer = final_answer_content
        
        end_total_process_mcp = time.perf_counter()
        response_trace.append(f"Timing - Total process_model_context_query duration: {end_total_process_mcp - start_total_process_mcp:.4f} seconds")

        return {
            "final_answer": final_answer,
            "trace": response_trace,
            "confidence_score": average_conf,
            "sources_used": {
                "local_sources_count": len(unique_local_filenames), # Count unique local files
                "local_files": unique_local_filenames, # Pass unique filenames
                "web_sources_count": len(sources_used['web_sources']), # Still count distinct web queries
                "web_links": unique_web_sources,
                "image_links": unique_image_links, # NEW: Include unique image links
                "used_local": len(unique_local_filenames) > 0 or len(unique_image_links) > 0,
                "used_web": len(unique_web_sources) > 0
            }
        }
    except Exception as e:
        logger.error(f"Error in process_model_context_query: {e}", exc_info=True)
        error_message = "An unexpected error occurred while processing your request."
        response_trace.append(f"ERROR: {e}")
        response_trace.append(traceback.format_exc())
        end_total_process_mcp = time.perf_counter()
        response_trace.append(f"Timing - Total process_model_context_query duration (Error): {end_total_process_mcp - start_total_process_mcp:.4f} seconds")
        return {
            "final_answer": error_message,
            "trace": response_trace,
            "confidence_score": 0.0,
            "sources_used": {
                "local_sources_count": 0,
                "local_files": [],
                "web_sources_count": 0,
                "web_links": [],
                "image_links": [],
                "used_local": False,
                "used_web": False
            }
        }

def main():
    """
    Main function to set up and run the enhanced hybrid chatbot.
    Handles CLI interaction, RAG strategy selection, and RAC toggling.
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
    logger.info("Creating VectorStoreIndex for local PDF data...")
    try:
        local_index = VectorStoreIndex.from_documents(
            documents,
            llm=OPTIMAL_LLM,
            embed_model=Settings.embed_model,
        )
        # Configure the query engine to include source nodes
        local_query_engine = local_index.as_query_engine(
            llm=OPTIMAL_LLM,
            # Important: The retriever must return source nodes for image/file extraction!
            response_mode="tree_summarize", 
            similarity_top_k=5,
        )
        # Also create a retriever instance for direct node retrieval
        local_retriever = local_index.as_retriever(similarity_top_k=5)

        logger.info("Local PDF data indexed successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not create VectorStoreIndex: {e}. Ensure Ollama models are running. Exiting.")
        sys.exit(1)

    # Validate Google API keys
    google_api_key, google_cse_id = validate_google_api_keys_from_env()
    if not (google_api_key and google_cse_id):
        logger.critical("FATAL ERROR: Google API keys not configured. Exiting.")
        sys.exit(1)
    
    # Initialize Google Custom Search Tool
    Google_Search_instance = GoogleCustomSearchTool(
        api_key=google_api_key,
        cse_id=google_cse_id,
        num_results=5 # Number of web search results to retrieve
    )

    # Initialize RAC Corrector
    rac_corrector = RACCorrector(
        llm=OPTIMAL_LLM,
        local_query_engine=local_query_engine,
        Google_Search_tool=Google_Search_instance
    )
    rac_corrector.testing_mode = testing_mode_enabled
    
    # Define Pydantic input schema for local RAG tool
    class LocalBookQAToolInput(BaseModel):
        query: str = Field(description="The question to ask about the PDF book content.")

    # Create LlamaIndex FunctionTool for local RAG
    # MODIFIED: This function now returns a JSON string containing the text, files, and images.
    def local_book_qa_function(query: str) -> str:
        """
        Function to expose local PDF querying to the LlamaIndex agent.
        Returns a JSON string containing the text response, local files, and image metadata.
        """
        logger.info(f"Local RAG: Querying for '{query}'")
        start_local_rag_query = time.perf_counter()
        
        # 1. Retrieve nodes first (using the retriever)
        nodes = local_retriever.retrieve(query)

        # 2. Extract images/files from nodes
        image_results = []
        unique_filenames = set()
        
        for node in nodes:
            # Add file to unique list
            filename = node.metadata.get('filename')
            if filename:
                 unique_filenames.add(filename)

            # --- NEW IMAGE LOGIC ADDITION ---
            name = node.metadata.get("name")
            caption = node.metadata.get("caption")
            page = node.metadata.get("page")
            if name and name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_results.append({
                    "url": SUPABASE_URL + name,
                    "caption": caption if caption else "Figure/Table from document",
                    "page": page,
                    "source_file": filename # Add source filename for traceability
                })
        
        # 3. Use Query Engine to synthesize the text response
        response_obj = local_query_engine.query(query) # This synthesizes the answer from the retrieved nodes
        response_text = str(response_obj)

        end_local_rag_query = time.perf_counter()
        logger.info(f"Timing - Local RAG Query: {end_local_rag_query - start_local_rag_query:.4f} seconds")
        
        # 4. Return packaged JSON string
        return json.dumps({
            "text": response_text,
            "local_files": sorted(list(unique_filenames)),
            "images": image_results
        })

    local_rag_tool = FunctionTool.from_defaults(
        fn=local_book_qa_function,
        name="local_book_qa",
        description=(
            "Useful for questions specifically about the content of the provided PDF book. "
            "The function returns a JSON string containing the text answer, source file names, and image URLs if figures/tables are relevant."
        ),
        fn_schema=LocalBookQAToolInput,
    )

    # Create LlamaIndex FunctionTool for Google Web Search
    Google_Search_tool_for_agent = FunctionTool.from_defaults(
        fn=Google_Search_instance.search_legacy,  
        name="google_web_search",
        description=(
            "Useful for general knowledge questions, current events, or anything requiring internet search."
        ),
    )

    # Combine all tools that the main agent can use
    tools_for_agent = [local_rag_tool, Google_Search_tool_for_agent]

    logger.info(f"Initializing ReAct Agent with LLM: {OPTIMAL_LLM_MODEL_NAME}...")
    # Initialize the main ReAct Agent that will orchestrate the responses
    agent = ReActAgent.from_tools(
        llm=OPTIMAL_LLM,
        tools=tools_for_agent,
        verbose=False, # Set to True for detailed agent internal steps (useful for debugging)
        max_iterations=30 # Limit agent's thinking iterations
    )
    
    logger.info("Initialized Enhanced ReAct Agent with RAC.")
    logger.info("\n--- Enhanced Hybrid Chatbot with Model Context Protocol READY ---")
    logger.info(f"Agent uses LLM: {OPTIMAL_LLM_MODEL_NAME}")
    logger.info(f"Tools available: {', '.join([t.metadata.name for t in tools_for_agent])}")
    logger.info(f"RAC enabled ({'Testing Mode' if testing_mode_enabled else 'Active'})")
    logger.info(f"Type your questions. Type 'exit' to quit.")
    logger.info(f"Commands: 'toggle_rac', 'rac_stats', 'set_mode [local|web|hybrid|automatic]'")
    
    # Initialize RAC to enabled by default
    rac_corrector.rac_enabled = True
    # Statistics for RAC performance tracking
    rac_stats = {
        'total_queries': 0,
        'corrected_queries': 0,
        'total_corrections': 0,
        'uncertain_claims_flagged': 0,
        'responses_suppressed': 0,
        'responses_flagged_low_confidence': 0
    }
    
    # Confidence thresholds for response handling
    SUPPRESS_THRESHOLD = 0.4 # Below this, response is suppressed
    FLAG_THRESHOLD = 0.6     # Below this, response is flagged for low confidence

    # Available RAG strategies
    RAG_STRATEGIES = {
        "1": "rac_enhanced_hybrid_rag", # Default, uses planning workflow + RAC
        "2": "planning_workflow",
        "3": "multi_step_query_engine",
        "4": "multi_strategy_workflow",
        "5": "no_method" # Pure agent chat, relying on its internal reasoning for tool use
    }
    STRATEGY_NAMES = {
        "rac_enhanced_hybrid_rag": "RAC Enhanced Hybrid RAG",
        "planning_workflow": "Planning Workflow",
        "multi_step_query_engine": "Multi-Step Query Engine",
        "multi_strategy_workflow": "Multi-Strategy Workflow",
        "no_method": "No Specific RAG Method"
    }
    current_rag_strategy = "1" # Default strategy on startup

    # Available retrieval methods for RAC and RAG strategies
    RETRIEVAL_METHODS = {
        "local": "Local Only (PDF)",
        "web": "Web Only (Google Search)",
        "hybrid": "Local and Web",
        "automatic": "Automatic" # Heuristic-based selection
    }
    current_retrieval_method = "hybrid" # Default retrieval method on startup
    
    context_memory = {} # Simple in-memory context for chat history (not persisted)
    
    # Main conversational loop
    while True:
        print("\n" + "="*50)
        print("Select RAG Strategy:")
        for key, value in RAG_STRATEGIES.items():
            print(f"  {key}. {STRATEGY_NAMES[value]}")
        strategy_choice = input(f"Enter strategy number (currently using {STRATEGY_NAMES[RAG_STRATEGIES[current_rag_strategy]]}): ").strip().lower()
        
        # Handle commands or strategy selection
        if strategy_choice == 'exit':
            logger.info("Exiting Enhanced Hybrid Chatbot. Goodbye!")
            break
        elif strategy_choice == 'toggle_rac':
            rac_corrector.rac_enabled = not rac_corrector.rac_enabled
            print(f"RAC is now {'ENABLED' if rac_corrector.rac_enabled else 'DISABLED'}")
            continue
        elif strategy_choice == 'rac_stats':
            # Display RAC statistics
            print(f"\n--- RAC Statistics ---")
            print(f"Total queries processed: {rac_stats['total_queries']}")
            print(f"Queries with corrections: {rac_stats['corrected_queries']}")
            print(f"Total corrections made: {rac_stats['total_corrections']}")
            print(f"Total uncertain claims flagged: {rac_stats['uncertain_claims_flagged']}")
            print(f"Responses suppressed: {rac_stats['responses_suppressed']}")
            print(f"Responses flagged: {rac_stats['responses_flagged_low_confidence']}")
            correction_rate = rac_stats['corrected_queries'] / max(rac_stats['total_queries'], 1) * 100
            print(f"Correction rate: {correction_rate:.1f}%")
            continue
        elif strategy_choice.startswith('set_mode'):
            # Change retrieval method
            parts = strategy_choice.split()
            if len(parts) == 2 and parts[0] == 'set_mode':
                mode = parts[1]
                if mode in RETRIEVAL_METHODS:
                    current_retrieval_method = mode
                    print(f"Retrieval method set to: {mode.upper()}.")
                else:
                    print(f"Invalid mode. Use 'set_mode {list(RETRIEVAL_METHODS.keys())}'.")
            else:
                print("Invalid command format. Use 'set_mode [local|web|hybrid|automatic]'.")
            continue
        elif strategy_choice in RAG_STRATEGIES:
            current_rag_strategy = strategy_choice
            print(f"Selected strategy: {STRATEGY_NAMES[RAG_STRATEGIES[current_rag_strategy]]}")
        else:
            print("Invalid strategy selection.")
            continue

        print("\n" + "="*50)
        print("Select Retrieval Method:")
        for key, desc in RETRIEVAL_METHODS.items():
            print(f"  {key}: {desc}")
        retrieval_choice = input(f"Enter retrieval method (currently using '{current_retrieval_method}'): ").strip().lower()
        
        if retrieval_choice in RETRIEVAL_METHODS:
            current_retrieval_method = retrieval_choice
            print(f"Selected retrieval method: {RETRIEVAL_METHODS[current_retrieval_method]}")
        else:
            print("Invalid retrieval method selected. Keeping current method.")
        
        user_question = input("Enter your question: ").strip()
        print("\n" + "="*50)
        print("--- Agent's Response (via Model Context Protocol) ---")
        print("="*50)

        # Build tools list for the current query based on selected retrieval method
        tools_for_query = []
        if current_retrieval_method in ["local", "hybrid", "automatic"]:
            tools_for_query.append(local_rag_tool)
        if current_retrieval_method in ["web", "hybrid", "automatic"]:
            tools_for_query.append(Google_Search_tool_for_agent)
        
        if not tools_for_query:
            print("❌ No retrieval source selected. Please choose a method that includes a data source (e.g., 'local', 'web', 'hybrid', 'automatic').")
            continue

        try:
            start_time = time.time()
            # Call the main processing function
            mcp_response = asyncio.run(process_model_context_query(
                query=user_question,
                context_memory=context_memory, # Pass current chat context
                tool_outputs=[], # Placeholder, populated internally by MCP
                scratchpad="",    # Placeholder, populated internally by MCP
                agent_instance=agent,
                rac_corrector_instance=rac_corrector,
                testing_mode=testing_mode_enabled,
                suppress_threshold=SUPPRESS_THRESHOLD,
                flag_threshold=FLAG_THRESHOLD,
                selected_rag_strategy=RAG_STRATEGIES[current_rag_strategy],
                selected_retrieval_method=current_retrieval_method,
                local_query_engine=local_query_engine,
                google_custom_search_instance=Google_Search_instance,
                tools_for_agent=tools_for_query # Pass filtered tools
            ))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n⏱️ Total processing time: {elapsed_time:.2f} seconds")
            
            # Update RAC statistics
            rac_stats['total_queries'] += 1
            if rac_corrector.rac_enabled:
                if "❌" in mcp_response["final_answer"]:
                    rac_stats['responses_suppressed'] += 1
                elif "⚠️" in mcp_response["final_answer"]:
                    rac_stats['responses_flagged_low_confidence'] += 1
            
            # Print the final answer
            print(mcp_response["final_answer"])
            
            # Display formatted sources information
            sources_info_str = format_sources_info(mcp_response["sources_used"])
            print(sources_info_str)

            # Print the detailed model context trace
            print("\n--- Model Context Trace ---")
            for step in mcp_response["trace"]:
                print(step)
            print("--- End Model Context Trace ---")

            # Update more RAC statistics based on trace content
            if rac_corrector.rac_enabled:
                if any("Corrections Applied:" in step for step in mcp_response["trace"]):
                    rac_stats['corrected_queries'] += 1
                if any("Uncertain Claims Flagged:" in step for step in mcp_response["trace"]):
                    rac_stats['uncertain_claims_flagged'] += 1

            # Store the current interaction in context memory
            context_memory[user_question] = mcp_response["final_answer"]
        except Exception as e:
            logger.error(f"Error during interaction loop: {e}", exc_info=True)
            print("An unhandled error occurred while processing your request. Please check logs for details.")
            
        print("="*50 + "\n")
    logger.info("Enhanced Hybrid Chatbot with Model Context Protocol Session Completed.")

def format_sources_info(sources_info: Dict[str, Any]) -> str:
    """
    Formats the sources information into a user-friendly string for display.
    Now includes specific local book names and image links.
    """
    if not sources_info:
        return "\n📚 **Sources Used:** None"
    
    info_lines = ["\n📚 **Sources Used:**"]
    
    local_files = sources_info.get('local_files', [])
    image_links = sources_info.get('image_links', []) # NEW

    if sources_info.get('used_local', False):
        if local_files:
            info_lines.append(f"  📄 **Local PDF Documents Referenced:**")
            for i, filename in enumerate(local_files[:5], 1): # Limit to top 5 for brevity
                info_lines.append(f"    {i}. {filename}")
            if len(local_files) > 5:
                info_lines.append(f"    ... and {len(local_files) - 5} more local files (not displayed)")
        
        if image_links: # NEW: Display image links
            info_lines.append(f"  🖼️ **Referenced Visual Aids/Images (Top 3):**")
            for i, link in enumerate(image_links[:3], 1):
                caption = link.get('caption', 'No caption available')
                page_info = f" (Page: {link['page']})" if link.get('page') else ""
                file_info = f" (Source: {link['source_file']})" if link.get('source_file') else ""
                info_lines.append(f"    {i}. **{caption}**{page_info}{file_info}")
                info_lines.append(f"       URL: {link['url']}")
            if len(image_links) > 3:
                info_lines.append(f"    ... and {len(image_links) - 3} more images (not displayed)")

    if sources_info.get('used_web', False):
        info_lines.append(f"  🌐 Web Search: {sources_info['web_sources_count']} queries")
        
        web_links = sources_info.get('web_links', [])
        if web_links:
            info_lines.append(f"\n🔗 **Web Sources Referenced (Top 5):**")
            for i, link in enumerate(web_links[:5], 1):  # Limit to top 5 links for brevity
                title = link.get('title', 'Unknown Title')
                url = link.get('url', '')
                snippet = link.get('snippet', '')
                
                info_lines.append(f"  {i}. **{title}**")
                info_lines.append(f"       URL: {url}")
                if snippet:
                    info_lines.append(f"       Preview: {snippet}")
                info_lines.append("") # Add a blank line for readability
            
            if len(web_links) > 5:
                info_lines.append(f"  ... and {len(web_links) - 5} more web sources (not displayed)")
    
    if not sources_info.get('used_local', False) and not sources_info.get('used_web', False):
        info_lines.append("  ℹ️ No external sources were consulted.")
    
    return "\n".join(info_lines)

# Entry point of the script
if __name__ == "__main__":
    main()