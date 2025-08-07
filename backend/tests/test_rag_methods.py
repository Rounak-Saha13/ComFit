"""
Comprehensive tests for RAG methods and retrieval strategies
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Import the modules to test
from chat_engine.RAG_methods import (
    FactualClaimExtractor,
    FactVerifier,
    RACCorrector,
    GoogleCustomSearchTool,
    run_planning_workflow,
    run_multi_step_query_engine_workflow,
    run_multi_strategy_workflow,
    process_model_context_query,
    clean_text,
    curate_pdf_to_text,
    load_single_document_for_indexing
)


class TestFactualClaimExtractor:
    """Test the FactualClaimExtractor class"""
    
    def test_init(self, mock_llm):
        """Test FactualClaimExtractor initialization"""
        extractor = FactualClaimExtractor(mock_llm)
        assert extractor.llm == mock_llm
    
    def test_extract_claims_success(self, mock_llm, sample_text_with_claims):
        """Test successful claim extraction"""
        # Mock LLM response with properly formatted claims
        mock_response = """
        CLAIM: The human heart has four chambers
        CLAIM: Water boils at 100 degrees Celsius at sea level
        CLAIM: The capital of France is Paris
        CLAIM: Python was first released in 1991
        """
        mock_llm.complete.return_value = mock_response
        
        extractor = FactualClaimExtractor(mock_llm)
        claims = extractor.extract_claims(sample_text_with_claims)
        
        assert len(claims) == 4
        assert "The human heart has four chambers" in claims
        assert "Water boils at 100 degrees Celsius at sea level" in claims
        assert "The capital of France is Paris" in claims
        assert "Python was first released in 1991" in claims
        
        # Verify LLM was called with extraction prompt
        mock_llm.complete.assert_called_once()
        call_args = mock_llm.complete.call_args[0][0]
        assert "Extract atomic factual claims" in call_args
        assert sample_text_with_claims in call_args
    
    def test_extract_claims_empty_response(self, mock_llm):
        """Test claim extraction with empty LLM response"""
        mock_llm.complete.return_value = ""
        
        extractor = FactualClaimExtractor(mock_llm)
        claims = extractor.extract_claims("Some text")
        
        assert claims == []
    
    def test_extract_claims_no_valid_claims(self, mock_llm):
        """Test claim extraction when no valid claims are found"""
        mock_response = """
        This is not a claim format
        Another line without proper format
        CLAIM: Short  # Too short claim
        """
        mock_llm.complete.return_value = mock_response
        
        extractor = FactualClaimExtractor(mock_llm)
        claims = extractor.extract_claims("Some text")
        
        assert claims == []
    
    def test_extract_claims_llm_error(self, mock_llm):
        """Test claim extraction when LLM throws an error"""
        mock_llm.complete.side_effect = Exception("LLM error")
        
        extractor = FactualClaimExtractor(mock_llm)
        claims = extractor.extract_claims("Some text")
        
        assert claims == []


class TestFactVerifier:
    """Test the FactVerifier class"""
    
    def test_init(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test FactVerifier initialization"""
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        assert verifier.llm == mock_llm
        assert verifier.local_query_engine == mock_local_query_engine
        assert verifier.Google_Search_tool == mock_google_search_tool
        assert verifier.verification_cache == {}
        assert verifier.reversal_min_confidence == 0.95
    
    def test_get_claim_hash(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test claim hash generation"""
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        claim1 = "The capital of France is Paris"
        claim2 = "THE CAPITAL OF FRANCE IS PARIS"  # Different case
        claim3 = "The capital of Germany is Berlin"  # Different content
        
        hash1 = verifier._get_claim_hash(claim1)
        hash2 = verifier._get_claim_hash(claim2)
        hash3 = verifier._get_claim_hash(claim3)
        
        # Same content (different case) should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        assert len(hash1) == 32  # MD5 hash length
    
    def test_extract_search_terms(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test search term extraction"""
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        claim = "The capital of France is Paris and it has many museums"
        search_terms = verifier._extract_search_terms(claim)
        
        # Should extract key words, exclude stop words
        assert "capital" in search_terms
        assert "France" in search_terms
        assert "Paris" in search_terms
        assert "museums" in search_terms
        assert "the" not in search_terms  # Stop word
        assert "is" not in search_terms   # Stop word
        assert "and" not in search_terms  # Stop word
    
    @patch('time.perf_counter')
    def test_verify_claim_local_evidence(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test claim verification with local evidence"""
        mock_time.return_value = 1.0
        
        # Mock local query engine response
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Paris is indeed the capital of France"
        
        # Mock LLM analysis response
        mock_llm.complete.return_value = """
        VERDICT: SUPPORTED
        CONFIDENCE: 0.9
        CORRECTION: None
        REASONING: Local evidence clearly supports the claim
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        result = verifier.verify_claim("The capital of France is Paris", use_local=True, use_web=False)
        
        assert result['is_supported'] == True
        assert result['confidence'] >= 0.9  # Should be boosted for local evidence
        assert result['correction_suggestion'] is None
        assert len(result['evidence']) == 1
        assert result['evidence'][0]['source'] == 'local_knowledge'
    
    @patch('time.perf_counter')
    def test_verify_claim_web_evidence(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test claim verification with web evidence only"""
        mock_time.return_value = 1.0
        
        # Mock local query engine to return no useful info
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "I don't know"
        
        # Mock web search response
        mock_google_search_tool.search.return_value = """
        Result 1: Title: France Capital
        Snippet: Paris is the capital and largest city of France
        Link: https://example.com/france
        """
        
        # Mock LLM analysis response
        mock_llm.complete.return_value = """
        VERDICT: SUPPORTED
        CONFIDENCE: 0.8
        CORRECTION: None
        REASONING: Web evidence supports the claim
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        result = verifier.verify_claim("The capital of France is Paris", use_local=True, use_web=True)
        
        assert result['is_supported'] == True
        assert result['confidence'] == 0.8
        assert result['correction_suggestion'] is None
        assert len(result['evidence']) == 1
        assert result['evidence'][0]['source'] == 'web_search'
    
    def test_verify_claim_cached(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test claim verification with cached result"""
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        # Pre-populate cache
        claim = "Test claim"
        cached_result = {
            'claim': claim,
            'is_supported': True,
            'confidence': 0.9,
            'evidence': [],
            'correction_suggestion': None,
            'warning': None
        }
        claim_hash = verifier._get_claim_hash(claim)
        verifier.verification_cache[claim_hash] = cached_result
        
        result = verifier.verify_claim(claim)
        
        # Should return cached result without calling LLM or engines
        assert result == cached_result
        mock_llm.complete.assert_not_called()
        mock_local_query_engine.query.assert_not_called()
        mock_google_search_tool.search.assert_not_called()
    
    @patch('time.perf_counter')
    def test_verify_claim_contradiction(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test claim verification with contradictory evidence"""
        mock_time.return_value = 1.0
        
        # Mock local evidence that contradicts the claim
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Berlin is the capital of Germany, not Paris"
        
        # Mock LLM analysis response showing contradiction
        mock_llm.complete.return_value = """
        VERDICT: CONTRADICTED
        CONFIDENCE: 0.9
        CORRECTION: The capital of Germany is Berlin
        REASONING: Local evidence contradicts the claim
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        result = verifier.verify_claim("The capital of Germany is Paris", use_local=True, use_web=False)
        
        assert result['is_supported'] == False
        assert result['confidence'] >= 0.9
        assert result['correction_suggestion'] == "The capital of Germany is Berlin"
    
    def test_verify_claim_no_evidence(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test claim verification with no evidence"""
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        result = verifier.verify_claim("Test claim", use_local=False, use_web=False)
        
        assert result['is_supported'] == False
        assert result['confidence'] == 0.0
        assert result['correction_suggestion'] is None
        assert result['warning'] == "No evidence found to verify this claim."


class TestRACCorrector:
    """Test the RACCorrector class"""
    
    def test_init(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test RACCorrector initialization"""
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        assert corrector.llm == mock_llm
        assert isinstance(corrector.claim_extractor, FactualClaimExtractor)
        assert isinstance(corrector.fact_verifier, FactVerifier)
        assert corrector.correction_threshold == 0.5
        assert corrector.uncertainty_threshold == 0.6
        assert corrector.verification_mode == "hybrid"
        assert corrector.rac_enabled == True
    
    def test_correct_response_rac_disabled(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test response correction when RAC is disabled"""
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        corrector.rac_enabled = False
        
        original_response = "Test response"
        result = corrector.correct_response(original_response)
        
        assert result['original_response'] == original_response
        assert result['corrected_response'] == original_response
        assert result['claims_analyzed'] == 0
        assert result['corrections_made'] == 0
        assert result['average_confidence'] == 1.0
    
    @patch('time.perf_counter')
    def test_correct_response_no_claims(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test response correction when no claims are extracted"""
        mock_time.return_value = 1.0
        
        # Mock claim extractor to return no claims
        mock_llm.complete.return_value = "No claims found"
        
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        original_response = "Hello, how are you?"
        result = corrector.correct_response(original_response)
        
        assert result['original_response'] == original_response
        assert result['corrected_response'] == original_response
        assert result['claims_analyzed'] == 0
        assert result['corrections_made'] == 0
    
    @patch('time.perf_counter')
    def test_correct_response_with_corrections(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test response correction with actual corrections needed"""
        mock_time.return_value = 1.0
        
        # Mock claim extraction
        mock_llm.complete.side_effect = [
            # First call: claim extraction
            "CLAIM: The capital of Germany is Paris",
            # Second call: evidence analysis
            """
            VERDICT: CONTRADICTED
            CONFIDENCE: 0.9
            CORRECTION: The capital of Germany is Berlin
            REASONING: Evidence shows Berlin is the capital
            """,
            # Third call: applying corrections
            "The capital of Germany is Berlin."
        ]
        
        # Mock local query engine
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Berlin is the capital of Germany"
        
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        original_response = "The capital of Germany is Paris."
        result = corrector.correct_response(original_response, apply_corrections=True)
        
        assert result['original_response'] == original_response
        assert result['corrected_response'] == "The capital of Germany is Berlin."
        assert result['claims_analyzed'] == 1
        assert result['corrections_made'] == 1
        assert len(result['corrections_applied']) == 1
        assert result['corrections_applied'][0]['correction'] == "The capital of Germany is Berlin"
    
    @patch('time.perf_counter')
    def test_correct_response_testing_mode(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test response correction in testing mode (no corrections applied)"""
        mock_time.return_value = 1.0
        
        # Mock claim extraction and verification
        mock_llm.complete.side_effect = [
            "CLAIM: The capital of Germany is Paris",
            """
            VERDICT: CONTRADICTED
            CONFIDENCE: 0.9
            CORRECTION: The capital of Germany is Berlin
            REASONING: Evidence shows Berlin is the capital
            """
        ]
        
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Berlin is the capital of Germany"
        
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        corrector.testing_mode = True
        
        original_response = "The capital of Germany is Paris."
        result = corrector.correct_response(original_response, apply_corrections=True)
        
        # In testing mode, corrections should be identified but not applied
        assert result['original_response'] == original_response
        assert result['corrected_response'] == original_response  # Not corrected in testing mode
        assert result['claims_analyzed'] == 1
        assert result['corrections_made'] == 1


class TestGoogleCustomSearchTool:
    """Test the GoogleCustomSearchTool class"""
    
    def test_init(self):
        """Test GoogleCustomSearchTool initialization"""
        tool = GoogleCustomSearchTool("test_api_key", "test_cse_id", num_results=5)
        
        assert tool.api_key == "test_api_key"
        assert tool.cse_id == "test_cse_id"
        assert tool.num_results == 5
        assert tool.base_url == "https://www.googleapis.com/customsearch/v1"
    
    @patch('requests.get')
    @patch('time.perf_counter')
    def test_search_success(self, mock_time, mock_get):
        """Test successful Google search"""
        mock_time.return_value = 1.0
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Test Result 1",
                    "snippet": "This is a test snippet 1",
                    "link": "https://example1.com"
                },
                {
                    "title": "Test Result 2", 
                    "snippet": "This is a test snippet 2",
                    "link": "https://example2.com"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        tool = GoogleCustomSearchTool("test_api_key", "test_cse_id")
        result = tool.search("test query")
        
        assert "Result 1: Title: Test Result 1" in result
        assert "Snippet: This is a test snippet 1" in result
        assert "Link: https://example1.com" in result
        assert "Result 2: Title: Test Result 2" in result
        
        # Verify API was called with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://www.googleapis.com/customsearch/v1"
        assert call_args[1]['params']['key'] == "test_api_key"
        assert call_args[1]['params']['cx'] == "test_cse_id"
        assert call_args[1]['params']['q'] == "test query"
    
    @patch('requests.get')
    @patch('time.perf_counter')
    def test_search_no_results(self, mock_time, mock_get):
        """Test Google search with no results"""
        mock_time.return_value = 1.0
        
        # Mock API response with no items
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {}  # No 'items' key
        mock_get.return_value = mock_response
        
        tool = GoogleCustomSearchTool("test_api_key", "test_cse_id")
        result = tool.search("test query")
        
        assert result == "No relevant search results found."
    
    @patch('requests.get')
    @patch('time.perf_counter')
    def test_search_api_error(self, mock_time, mock_get):
        """Test Google search with API error"""
        mock_time.return_value = 1.0
        
        # Mock API error
        mock_get.side_effect = Exception("API Error")
        
        tool = GoogleCustomSearchTool("test_api_key", "test_cse_id")
        result = tool.search("test query")
        
        assert "Error performing web search" in result
        assert "API Error" in result


class TestRAGStrategies:
    """Test RAG strategy functions"""
    
    @pytest.mark.asyncio
    async def test_run_planning_workflow(self, mock_react_agent):
        """Test planning workflow execution"""
        mock_react_agent.chat.return_value = Mock(response="Planning workflow response")
        
        trace = []
        result = await run_planning_workflow("test query", mock_react_agent, trace)
        
        assert result == "Planning workflow response"
        assert len(trace) >= 1
        assert "Planning Workflow" in trace[0]
        mock_react_agent.chat.assert_called_once_with("test query")
    
    @pytest.mark.asyncio
    async def test_run_planning_workflow_error(self, mock_react_agent):
        """Test planning workflow with error"""
        mock_react_agent.chat.side_effect = Exception("Agent error")
        
        trace = []
        result = await run_planning_workflow("test query", mock_react_agent, trace)
        
        assert "error occurred during the planning workflow" in result
        assert any("Error in Planning Workflow" in step for step in trace)
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_run_multi_step_query_engine_workflow(self, mock_to_thread, mock_local_query_engine, mock_google_search_tool):
        """Test multi-step query engine workflow"""
        # Mock asyncio.to_thread calls
        mock_to_thread.side_effect = [
            Mock(response="Multi-step response"),  # Router query engine response
        ]
        
        trace = []
        result = await run_multi_step_query_engine_workflow(
            "test query", mock_local_query_engine, mock_google_search_tool, trace
        )
        
        assert result == "Multi-step response"
        assert len(trace) >= 1
        assert "Multi-Step Query Engine" in trace[0]
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_run_multi_strategy_workflow(self, mock_to_thread, mock_local_query_engine, mock_google_search_tool):
        """Test multi-strategy workflow"""
        # Mock asyncio.to_thread calls for local and web queries, then synthesis
        mock_to_thread.side_effect = [
            Mock(response="Local response"),     # Local query
            Mock(response="Web response"),       # Web query  
            Mock(response="Synthesized response")  # Final synthesis
        ]
        
        trace = []
        result = await run_multi_strategy_workflow(
            "test query", mock_local_query_engine, mock_google_search_tool, trace
        )
        
        assert result == "Synthesized response"
        assert len(trace) >= 1
        assert "Multi-Strategy Workflow" in trace[0]


class TestModelContextProcessing:
    """Test Model Context Protocol processing functions"""
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    @patch('time.perf_counter')
    async def test_process_model_context_query(self, mock_time, mock_to_thread, sample_context_memory, mock_react_agent):
        """Test model context query processing"""
        mock_time.return_value = 1.0
        
        # Mock RACCorrector
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = True
        mock_rac_corrector.verification_mode = "hybrid"
        mock_rac_corrector.correct_response.return_value = {
            'original_response': 'Original response',
            'corrected_response': 'Corrected response',
            'claims_analyzed': 2,
            'corrections_made': 1,
            'verification_results': [],
            'corrections_applied': [],
            'uncertain_claims': [],
            'average_confidence': 0.8
        }
        
        # Mock asyncio.to_thread calls
        mock_to_thread.side_effect = [
            Mock(response="Agent response"),  # Agent chat
            Mock(response="Final answer")     # Final LLM completion
        ]
        
        result = await process_model_context_query(
            query="test query",
            context_memory=sample_context_memory,
            tool_outputs=[],
            scratchpad="",
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,
            flag_threshold=0.6,
            selected_rag_strategy="rac_enhanced_hybrid_rag",
            local_query_engine=None,
            google_custom_search_instance=None
        )
        
        assert "final_answer" in result
        assert "trace" in result
        assert "confidence_score" in result
        assert result["confidence_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_process_model_context_query_error(self, sample_context_memory, mock_react_agent):
        """Test model context query processing with error"""
        # Mock agent to raise exception
        mock_react_agent.chat.side_effect = Exception("Agent error")
        
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = True
        mock_rac_corrector.verification_mode = "hybrid"
        
        result = await process_model_context_query(
            query="test query",
            context_memory=sample_context_memory,
            tool_outputs=[],
            scratchpad="",
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,
            flag_threshold=0.6,
            selected_rag_strategy="rac_enhanced_hybrid_rag",
            local_query_engine=None,
            google_custom_search_instance=None
        )
        
        assert "final_answer" in result
        assert "error occurred" in result["final_answer"]
        assert result["confidence_score"] == 0.0
        assert any("ERROR:" in step for step in result["trace"])


class TestPDFProcessing:
    """Test PDF processing functions"""
    
    def test_clean_text(self):
        """Test text cleaning function"""
        dirty_text = """This is a test-
        ing text with
        hyphens and line breaks.

        It has multiple    spaces    and
        page numbers like --- PAGE 5 ---
        and standalone numbers.
        123
        More text here."""
        
        cleaned = clean_text(dirty_text)
        
        # Should remove hyphenated line breaks
        assert "testing text" in cleaned
        # Should remove page markers
        assert "--- PAGE 5 ---" not in cleaned
        # Should normalize whitespace
        assert "multiple    spaces" not in cleaned
        assert "multiple spaces" in cleaned
    
    def test_load_single_document_for_indexing_missing_file(self):
        """Test loading document when file doesn't exist"""
        with pytest.raises(SystemExit):
            load_single_document_for_indexing("/nonexistent/file.txt")
    
    @patch('chat_engine.RAG_methods.SimpleDirectoryReader')
    def test_load_single_document_for_indexing_success(self, mock_reader, temp_text_file):
        """Test successful document loading"""
        # Mock SimpleDirectoryReader
        mock_doc = Mock()
        mock_doc.metadata = {}
        mock_reader_instance = Mock()
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader.return_value = mock_reader_instance
        
        documents = load_single_document_for_indexing(temp_text_file)
        
        assert len(documents) == 1
        assert documents[0].metadata['category'] == 'BookContent'
        assert documents[0].metadata['filename'] == os.path.basename(temp_text_file)
    
    @patch('chat_engine.RAG_methods.SimpleDirectoryReader')
    def test_load_single_document_for_indexing_no_content(self, mock_reader, temp_text_file):
        """Test document loading when no content is loaded"""
        # Mock SimpleDirectoryReader to return empty list
        mock_reader_instance = Mock()
        mock_reader_instance.load_data.return_value = []
        mock_reader.return_value = mock_reader_instance
        
        with pytest.raises(SystemExit):
            load_single_document_for_indexing(temp_text_file)


if __name__ == "__main__":
    pytest.main([__file__])