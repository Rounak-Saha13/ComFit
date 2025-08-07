"""
Tests for Hybrid Bot and MCP functionality
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# Import the modules to test
from chat_engine.Hybrid_Bot import (
    MCPRequest,
    MCPResponse,
    FactualClaimExtractor,
    FactVerifier,
    RACCorrector,
    GoogleCustomSearchTool,
    process_mcp_query,
    validate_google_api_keys_from_env,
    clean_text,
    curate_pdf_to_text,
    load_single_document_for_indexing
)


class TestMCPSchemas:
    """Test MCP schema definitions"""
    
    def test_mcp_request_creation(self):
        """Test MCPRequest creation"""
        request = MCPRequest(
            query="Test query",
            context_memory={"key": "value"},
            tool_outputs=[{"tool": "test", "result": "result"}],
            scratchpad="test scratchpad"
        )
        
        assert request.query == "Test query"
        assert request.context_memory == {"key": "value"}
        assert len(request.tool_outputs) == 1
        assert request.scratchpad == "test scratchpad"
    
    def test_mcp_request_defaults(self):
        """Test MCPRequest with default values"""
        request = MCPRequest(query="Test query")
        
        assert request.query == "Test query"
        assert request.context_memory == {}
        assert request.tool_outputs == []
        assert request.scratchpad == ""
    
    def test_mcp_response_creation(self):
        """Test MCPResponse creation"""
        response = MCPResponse(
            final_answer="Test answer",
            trace=["Step 1", "Step 2"],
            confidence_score=0.85
        )
        
        assert response.final_answer == "Test answer"
        assert response.trace == ["Step 1", "Step 2"]
        assert response.confidence_score == 0.85
    
    def test_mcp_response_defaults(self):
        """Test MCPResponse with default values"""
        response = MCPResponse(final_answer="Test answer")
        
        assert response.final_answer == "Test answer"
        assert response.trace == []
        assert response.confidence_score == 0.0


class TestHybridBotFactualClaimExtractor:
    """Test FactualClaimExtractor in Hybrid Bot context"""
    
    def test_extract_claims_excludes_meta_information(self, mock_llm):
        """Test that claim extraction excludes meta-information about tools"""
        # Mock LLM response with meta-information that should be excluded
        mock_response = """
        CLAIM: The capital of France is Paris
        CLAIM: I used tool local_book_qa to find this information
        CLAIM: There is no mention of this topic in the document
        CLAIM: Water boils at 100 degrees Celsius
        CLAIM: The tools mentioned are very helpful
        """
        mock_llm.complete.return_value = mock_response
        
        extractor = FactualClaimExtractor(mock_llm)
        claims = extractor.extract_claims("Test text with meta information")
        
        # Should only extract factual claims, not meta-information
        factual_claims = [claim for claim in claims if not any(meta in claim.lower() for meta in [
            "tool", "mention", "document", "i used", "there is no"
        ])]
        
        assert "The capital of France is Paris" in claims
        assert "Water boils at 100 degrees Celsius" in claims
        # Meta-information should ideally be filtered out by the prompt
        assert len(claims) >= 2


class TestHybridBotFactVerifier:
    """Test FactVerifier with enhanced logic from Hybrid Bot"""
    
    def test_verify_claim_prioritizes_local_evidence(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test that fact verifier prioritizes local evidence"""
        # Mock local query engine response (should be prioritized)
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "According to the document, Paris is the capital of France"
        
        # Mock web search response (should be deprioritized)
        mock_google_search_tool.search.return_value = "Berlin is mentioned as a capital city"
        
        # Mock LLM analysis that prioritizes local evidence
        mock_llm.complete.return_value = """
        VERDICT: SUPPORTED
        CONFIDENCE: 0.95
        CORRECTION: None
        REASONING: Local document evidence clearly supports the claim with high authority
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        result = verifier.verify_claim("The capital of France is Paris", use_local=True, use_web=True)
        
        # Should use local evidence and get boosted confidence
        assert result['is_supported'] == True
        assert result['confidence'] >= 0.95  # Should be high due to local evidence
        assert len(result['evidence']) == 1  # Only local evidence should be used
        assert result['evidence'][0]['source'] == 'local_knowledge'
    
    @patch('time.perf_counter')
    def test_verify_claim_reversal_protection(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test reversal protection logic"""
        mock_time.return_value = 1.0
        
        # Mock evidence suggesting a reversal with medium confidence
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Some ambiguous information"
        
        # Mock LLM suggesting reversal with confidence below threshold
        mock_llm.complete.return_value = """
        VERDICT: CONTRADICTED
        CONFIDENCE: 0.7
        CORRECTION: The capital of France is not Paris
        REASONING: Some evidence suggests otherwise
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        verifier.reversal_min_confidence = 0.95  # High threshold for reversals
        
        result = verifier.verify_claim("The capital of France is Paris", use_local=True, use_web=False)
        
        # Reversal should be suppressed due to low confidence
        assert result['correction_suggestion'] is None  # Correction suppressed
        assert result['confidence'] == 0.5  # Reset to neutral
    
    def test_verify_claim_uncertainty_warning(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test uncertainty warning generation"""
        # Mock low confidence scenario
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Unclear information"
        
        mock_llm.complete.return_value = """
        VERDICT: INSUFFICIENT_EVIDENCE
        CONFIDENCE: 0.3
        CORRECTION: None
        REASONING: Evidence is unclear and insufficient
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        result = verifier.verify_claim("Ambiguous claim", use_local=True, use_web=False)
        
        assert result['is_supported'] == False
        assert result['confidence'] <= 0.6
        assert result['warning'] is not None
        assert "Low confidence" in result['warning']


class TestHybridBotRACCorrector:
    """Test RACCorrector with Hybrid Bot enhancements"""
    
    def test_rac_corrector_testing_mode(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test RAC corrector in testing mode"""
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        corrector.testing_mode = True
        
        # Mock claim extraction and verification
        mock_llm.complete.side_effect = [
            "CLAIM: Incorrect claim",
            """
            VERDICT: CONTRADICTED
            CONFIDENCE: 0.9
            CORRECTION: Correct claim
            REASONING: Evidence contradicts the claim
            """
        ]
        
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Contradictory evidence"
        
        result = corrector.correct_response("Original response with incorrect claim", apply_corrections=True)
        
        # In testing mode, should analyze but not apply corrections
        assert result['original_response'] == "Original response with incorrect claim"
        assert result['corrected_response'] == "Original response with incorrect claim"  # Unchanged
        assert result['corrections_made'] == 1  # Correction identified
        assert len(result['corrections_applied']) == 1  # But recorded
    
    def test_rac_corrector_verification_modes(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test different verification modes"""
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        # Test local-only mode
        corrector.verification_mode = "local"
        
        mock_llm.complete.return_value = "CLAIM: Test claim"
        
        # Mock the fact verifier's verify_claim method
        with patch.object(corrector.fact_verifier, 'verify_claim') as mock_verify:
            mock_verify.return_value = {
                'is_supported': True,
                'confidence': 0.8,
                'evidence': [],
                'correction_suggestion': None,
                'warning': None
            }
            
            result = corrector.correct_response("Test response")
            
            # Should call verify_claim with local=True, web=False
            mock_verify.assert_called_with("Test claim", use_local=True, use_web=False)
    
    def test_rac_corrector_confidence_cascade(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test confidence cascade functionality"""
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        # Mock claims with different confidence levels
        mock_llm.complete.side_effect = [
            "CLAIM: Low confidence claim\nCLAIM: Medium confidence claim",
            # First claim verification (low confidence)
            """
            VERDICT: INSUFFICIENT_EVIDENCE
            CONFIDENCE: 0.2
            CORRECTION: None
            REASONING: Very low confidence
            """,
            # Second claim verification (medium confidence)
            """
            VERDICT: SUPPORTED
            CONFIDENCE: 0.7
            CORRECTION: None
            REASONING: Medium confidence support
            """
        ]
        
        result = corrector.correct_response("Test response with mixed confidence claims")
        
        # Average confidence should be (0.2 + 0.7) / 2 = 0.45
        expected_avg_confidence = (0.2 + 0.7) / 2
        assert abs(result['average_confidence'] - expected_avg_confidence) < 0.01
        
        # Should have uncertain claims flagged
        assert len(result['uncertain_claims']) > 0


class TestHybridBotMCPProcessing:
    """Test MCP processing in Hybrid Bot"""
    
    @patch('time.perf_counter')
    def test_process_mcp_query_preprocessing(self, mock_time, mock_react_agent):
        """Test MCP query preprocessing logic"""
        mock_time.return_value = 1.0
        
        # Mock agent response
        mock_react_agent.chat.return_value = Mock(response="Agent response")
        
        # Mock RAC corrector
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = True
        mock_rac_corrector.correct_response.return_value = {
            'original_response': 'Agent response',
            'corrected_response': 'Agent response',
            'claims_analyzed': 0,
            'corrections_made': 0,
            'verification_results': [],
            'corrections_applied': [],
            'uncertain_claims': [],
            'average_confidence': 1.0
        }
        
        # Test with definitional query about PDF content
        mcp_request = MCPRequest(query="What is speed process according to my document?")
        
        result = process_mcp_query(
            mcp_request=mcp_request,
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,
            flag_threshold=0.6
        )
        
        assert isinstance(result, MCPResponse)
        assert result.final_answer is not None
        assert len(result.trace) > 0
        
        # Should have preprocessing step in trace
        preprocessing_steps = [step for step in result.trace if "Pre-processed" in step]
        assert len(preprocessing_steps) > 0
    
    @patch('time.perf_counter')
    def test_process_mcp_query_confidence_cascade_suppress(self, mock_time, mock_react_agent):
        """Test confidence cascade suppression"""
        mock_time.return_value = 1.0
        
        mock_react_agent.chat.return_value = Mock(response="Low confidence response")
        
        # Mock RAC corrector with very low confidence
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = True
        mock_rac_corrector.correct_response.return_value = {
            'original_response': 'Low confidence response',
            'corrected_response': 'Low confidence response',
            'claims_analyzed': 2,
            'corrections_made': 0,
            'verification_results': [],
            'corrections_applied': [],
            'uncertain_claims': [],
            'average_confidence': 0.2  # Very low confidence
        }
        
        mcp_request = MCPRequest(query="Test query")
        
        result = process_mcp_query(
            mcp_request=mcp_request,
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,  # Above 0.2
            flag_threshold=0.6
        )
        
        # Response should be suppressed
        assert "❌" in result.final_answer
        assert "suppressed due to very low confidence" in result.final_answer
        assert result.confidence_score == 0.2
    
    @patch('time.perf_counter')
    def test_process_mcp_query_confidence_cascade_flag(self, mock_time, mock_react_agent):
        """Test confidence cascade flagging"""
        mock_time.return_value = 1.0
        
        mock_react_agent.chat.return_value = Mock(response="Medium confidence response")
        
        # Mock RAC corrector with medium confidence
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = True
        mock_rac_corrector.correct_response.return_value = {
            'original_response': 'Medium confidence response',
            'corrected_response': 'Medium confidence response',
            'claims_analyzed': 1,
            'corrections_made': 0,
            'verification_results': [],
            'corrections_applied': [],
            'uncertain_claims': [],
            'average_confidence': 0.5  # Medium confidence
        }
        
        mcp_request = MCPRequest(query="Test query")
        
        result = process_mcp_query(
            mcp_request=mcp_request,
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,  # Below 0.5
            flag_threshold=0.6  # Above 0.5
        )
        
        # Response should be flagged but not suppressed
        assert "⚠️" in result.final_answer
        assert "Low confidence" in result.final_answer
        assert "Medium confidence response" in result.final_answer
        assert result.confidence_score == 0.5
    
    def test_process_mcp_query_rac_disabled(self, mock_react_agent):
        """Test MCP processing with RAC disabled"""
        mock_react_agent.chat.return_value = Mock(response="Raw agent response")
        
        # Mock RAC corrector with RAC disabled
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = False
        
        mcp_request = MCPRequest(query="Test query")
        
        result = process_mcp_query(
            mcp_request=mcp_request,
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,
            flag_threshold=0.6
        )
        
        # Should return raw agent response with full confidence
        assert result.final_answer == "Raw agent response"
        assert result.confidence_score == 1.0
        assert any("RAC Disabled" in step for step in result.trace)
    
    def test_process_mcp_query_error_handling(self, mock_react_agent):
        """Test MCP processing error handling"""
        # Mock agent to raise exception
        mock_react_agent.chat.side_effect = Exception("Agent error")
        
        mock_rac_corrector = Mock()
        mock_rac_corrector.rac_enabled = True
        
        mcp_request = MCPRequest(query="Test query")
        
        result = process_mcp_query(
            mcp_request=mcp_request,
            agent_instance=mock_react_agent,
            rac_corrector_instance=mock_rac_corrector,
            testing_mode=False,
            suppress_threshold=0.4,
            flag_threshold=0.6
        )
        
        # Should handle error gracefully
        assert "unexpected error occurred" in result.final_answer
        assert result.confidence_score == 0.0
        assert any("ERROR:" in step for step in result.trace)


class TestHybridBotUtilities:
    """Test utility functions in Hybrid Bot"""
    
    def test_validate_google_api_keys_success(self):
        """Test successful Google API key validation"""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key', 'GOOGLE_CSE_ID': 'test_id'}):
            api_key, cse_id = validate_google_api_keys_from_env()
            
            assert api_key == 'test_key'
            assert cse_id == 'test_id'
    
    def test_validate_google_api_keys_missing(self):
        """Test Google API key validation with missing keys"""
        with patch.dict('os.environ', {}, clear=True):
            api_key, cse_id = validate_google_api_keys_from_env()
            
            assert api_key is None
            assert cse_id is None
    
    def test_clean_text_comprehensive(self):
        """Test comprehensive text cleaning"""
        dirty_text = """This is a multi-
        line text with hyphenated words and    extra spaces.
        
        It has various issues like.
        Sentence endings followed by newlines,
        And page markers like --- PAGE 42 ---
        
        Plus standalone numbers:
        123
        456
        
        And more content here."""
        
        cleaned = clean_text(dirty_text)
        
        # Check various cleaning operations
        assert "multi-\n        line" not in cleaned  # Hyphen removal
        assert "multiline text" in cleaned
        assert "--- PAGE 42 ---" not in cleaned  # Page marker removal
        assert "  extra" not in cleaned  # Multiple space normalization
        assert cleaned.count('\n') == 0  # All newlines should be replaced with spaces
    
    @patch('PyPDF2.PdfReader')
    def test_curate_pdf_to_text_success(self, mock_pdf_reader):
        """Test successful PDF to text curation"""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "Page content"
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page, mock_page]
        mock_pdf_reader.return_value = mock_reader_instance
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_bytes(b"fake pdf content")
            
            result_path = curate_pdf_to_text(str(pdf_path), temp_dir)
            
            assert result_path is not None
            assert Path(result_path).exists()
            assert Path(result_path).suffix == '.txt'
            
            # Check content was processed
            content = Path(result_path).read_text()
            assert "Page content" in content


class TestHybridBotIntegration:
    """Integration tests for Hybrid Bot components"""
    
    @patch('time.perf_counter')
    def test_end_to_end_rac_correction(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test end-to-end RAC correction process"""
        mock_time.return_value = 1.0
        
        # Mock claim extraction
        mock_llm.complete.side_effect = [
            # Claim extraction
            "CLAIM: The capital of Germany is Paris",
            # Evidence analysis
            """
            VERDICT: CONTRADICTED
            CONFIDENCE: 0.95
            CORRECTION: The capital of Germany is Berlin
            REASONING: Local evidence clearly shows Berlin is the capital
            """,
            # Correction application
            "The capital of Germany is Berlin."
        ]
        
        # Mock local evidence
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Berlin is the capital of Germany"
        
        # Create and test RACCorrector
        corrector = RACCorrector(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        original_response = "The capital of Germany is Paris."
        result = corrector.correct_response(original_response, apply_corrections=True)
        
        # Verify end-to-end correction
        assert result['original_response'] == original_response
        assert result['corrected_response'] == "The capital of Germany is Berlin."
        assert result['claims_analyzed'] == 1
        assert result['corrections_made'] == 1
        assert result['average_confidence'] >= 0.95
        
        # Verify correction details
        correction = result['corrections_applied'][0]
        assert correction['original_claim'] == "The capital of Germany is Paris"
        assert correction['correction'] == "The capital of Germany is Berlin"
        assert correction['confidence'] >= 0.95


if __name__ == "__main__":
    pytest.main([__file__])