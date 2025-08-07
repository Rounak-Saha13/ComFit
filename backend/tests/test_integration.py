"""
Integration tests for RAG methods and retrieval strategies
"""
import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Import modules for integration testing
from chat_engine.client import ChatEngine
from chat_engine.RAG_methods import (
    process_model_context_query,
    GoogleCustomSearchTool,
    RACCorrector,
    FactualClaimExtractor,
    FactVerifier
)
from chat_engine.Hybrid_Bot import (
    MCPRequest,
    MCPResponse,
    process_mcp_query
)


class TestRAGIntegration:
    """Integration tests for RAG system components"""
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_planning_workflow(self):
        """Test complete RAG pipeline with planning workflow"""
        engine = ChatEngine()
        
        # Mock all external dependencies
        with patch('ollama.chat') as mock_ollama_chat, \
             patch('llama_index.llms.ollama.Ollama') as mock_llm_class, \
             patch('llama_index.embeddings.ollama.OllamaEmbedding') as mock_embed_class, \
             patch('llama_index.core.agent.ReActAgent.from_tools') as mock_agent_class, \
             patch.object(engine, '_get_local_query_engine') as mock_get_query_engine:
            
            # Setup mocks
            mock_ollama_chat.return_value = {"message": {"content": "Ollama fallback response"}}
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            mock_query_engine = Mock()
            mock_query_engine.query.return_value = Mock(response="Local document response")
            mock_get_query_engine.return_value = mock_query_engine
            
            mock_agent = Mock()
            mock_agent.chat.return_value = Mock(response="Planning workflow executed successfully")
            mock_agent_class.return_value = mock_agent
            
            # Test messages
            messages = [
                {"sender": "user", "content": "What is anthropometry in the context of sizing?"}
            ]
            
            # Execute planning workflow
            result = await engine.generate_response(
                messages=messages,
                model="llama3:latest",
                rag_method="Planning Workflow",
                retrieval_method="local context only",
                preset="default"
            )
            
            # Verify results
            assert result["result"] == "Planning workflow executed successfully"
            assert "duration" in result
            assert result["ai_message"]["content"] == "Planning workflow executed successfully"
            
            # Verify agent was created with tools
            mock_agent_class.assert_called_once()
            call_args = mock_agent_class.call_args
            assert len(call_args[1]["tools"]) >= 1  # Should have at least local tool
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_multi_strategy(self):
        """Test complete RAG pipeline with multi-strategy workflow"""
        engine = ChatEngine()
        
        with patch('asyncio.to_thread') as mock_to_thread, \
             patch.object(engine, '_get_local_query_engine') as mock_get_query_engine, \
             patch('chat_engine.RAG_methods.GoogleCustomSearchTool') as mock_google_class:
            
            # Setup mocks
            mock_query_engine = Mock()
            mock_get_query_engine.return_value = mock_query_engine
            
            mock_google_tool = Mock()
            mock_google_class.return_value = mock_google_tool
            
            # Mock asyncio.to_thread calls for the multi-strategy workflow
            mock_to_thread.side_effect = [
                Mock(response="Local RAG response about sizing"),  # Local query
                Mock(response="Web search response about anthropometry"),  # Web query
                Mock(response="Combined response: Anthropometry is the measurement of human body dimensions, crucial for sizing in the garment industry.")  # Synthesis
            ]
            
            messages = [
                {"sender": "user", "content": "Explain anthropometry and its role in sizing"}
            ]
            
            # Execute multi-strategy workflow
            result = await engine.generate_response(
                messages=messages,
                model="llama3:latest",
                rag_method="Multi-Strategy Workflow",
                retrieval_method="Hybrid context",
                preset="CFIR"
            )
            
            # Verify results
            expected_response = "Combined response: Anthropometry is the measurement of human body dimensions, crucial for sizing in the garment industry."
            assert result["result"] == expected_response
            
            # Verify both local and web strategies were executed
            assert mock_to_thread.call_count == 3  # Local + Web + Synthesis
    
    @pytest.mark.asyncio
    async def test_rac_enhanced_integration(self):
        """Test RAC enhanced RAG integration"""
        engine = ChatEngine()
        
        with patch('chat_engine.RAG_methods.process_model_context_query') as mock_process_mcp, \
             patch.object(engine, '_get_local_query_engine') as mock_get_query_engine, \
             patch('chat_engine.RAG_methods.GoogleCustomSearchTool') as mock_google_class, \
             patch('chat_engine.RAG_methods.RACCorrector') as mock_rac_class:
            
            # Setup mocks
            mock_query_engine = Mock()
            mock_get_query_engine.return_value = mock_query_engine
            
            mock_google_tool = Mock()
            mock_google_class.return_value = mock_google_tool
            
            mock_rac_corrector = Mock()
            mock_rac_class.return_value = mock_rac_corrector
            
            # Mock process_model_context_query to return realistic result
            mock_process_mcp.return_value = {
                "final_answer": "✅ High confidence response: Anthropometry is the scientific study of human body measurements and proportions, essential for creating well-fitting garments.",
                "trace": [
                    "Processing RAC Enhanced Hybrid RAG",
                    "Claims analyzed: 2",
                    "Corrections made: 0", 
                    "Confidence: 0.9"
                ],
                "confidence_score": 0.9
            }
            
            messages = [
                {"sender": "user", "content": "What is anthropometry and why is it important for clothing fit?"}
            ]
            
            # Execute RAC enhanced workflow
            result = await engine.generate_response(
                messages=messages,
                model="llama3:latest",
                rag_method="RAC Enhanced Hybrid RAG",
                retrieval_method="Hybrid context",
                preset="CFIR"
            )
            
            # Verify results
            assert "High confidence response" in result["result"]
            assert "Anthropometry" in result["result"]
            
            # Verify RAC processing was called
            mock_process_mcp.assert_called_once()
            call_args = mock_process_mcp.call_args[1]
            assert call_args["selected_rag_strategy"] == "rac_enhanced_hybrid_rag"


class TestRAGPerformance:
    """Performance and stress tests for RAG components"""
    
    @pytest.mark.asyncio
    async def test_claim_extraction_performance(self, mock_llm):
        """Test claim extraction performance with large text"""
        # Create large text with multiple claims
        large_text = """
        Water boils at 100 degrees Celsius at sea level atmospheric pressure.
        The human heart has four chambers: two atria and two ventricles.
        Paris is the capital and largest city of France, located in the north-central part of the country.
        Python programming language was first released in 1991 by Guido van Rossum.
        The speed of light in vacuum is approximately 299,792,458 meters per second.
        Earth's circumference at the equator is approximately 40,075 kilometers.
        DNA stands for Deoxyribonucleic Acid and contains genetic instructions.
        The Great Wall of China is over 13,000 miles long in total length.
        Mount Everest is the highest mountain peak in the world at 8,848.86 meters.
        Shakespeare wrote approximately 37 plays and 154 sonnets during his career.
        """ * 10  # Multiply to create larger text
        
        # Mock LLM to return formatted claims
        mock_response = "\n".join([f"CLAIM: Claim {i}" for i in range(50)])
        mock_llm.complete.return_value = mock_response
        
        extractor = FactualClaimExtractor(mock_llm)
        
        # Measure performance
        start_time = time.time()
        claims = extractor.extract_claims(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert len(claims) > 0
        assert processing_time < 5.0  # Should complete within 5 seconds
        
        # Verify LLM was called efficiently (only once)
        assert mock_llm.complete.call_count == 1
    
    @pytest.mark.asyncio 
    async def test_verification_caching_performance(self, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test verification caching performance"""
        # Setup verifier
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        
        # Mock responses
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "Test evidence"
        
        mock_llm.complete.return_value = """
        VERDICT: SUPPORTED
        CONFIDENCE: 0.8
        CORRECTION: None
        REASONING: Evidence supports claim
        """
        
        test_claim = "Test claim for caching"
        
        # First call - should hit LLM and local engine
        start_time = time.time()
        result1 = verifier.verify_claim(test_claim)
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        result2 = verifier.verify_claim(test_claim)
        second_call_time = time.time() - start_time
        
        # Verify caching worked
        assert result1 == result2
        assert second_call_time < first_call_time  # Cache should be faster
        
        # Verify LLM was only called once (cached on second call)
        assert mock_llm.complete.call_count == 1
        assert mock_local_query_engine.query.call_count == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_rag_requests(self):
        """Test handling multiple concurrent RAG requests"""
        engine = ChatEngine()
        
        # Mock dependencies
        with patch.object(engine, '_process_default') as mock_process:
            mock_process.return_value = "Concurrent response"
            
            # Create multiple concurrent requests
            messages = [{"sender": "user", "content": f"Question {i}"}]
            
            # Execute concurrent requests
            tasks = []
            for i in range(10):
                task = engine.generate_response(
                    messages=[{"sender": "user", "content": f"Question {i}"}],
                    model="llama3:latest",
                    rag_method="No Specific RAG Method"
                )
                tasks.append(task)
            
            # Wait for all to complete
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Verify all requests completed successfully
            assert len(results) == 10
            assert all(result["result"] == "Concurrent response" for result in results)
            
            # Should complete within reasonable time (concurrent processing)
            assert total_time < 10.0


class TestRAGErrorHandling:
    """Test error handling and resilience in RAG components"""
    
    @pytest.mark.asyncio
    async def test_llm_connection_failure_resilience(self):
        """Test resilience when LLM connection fails"""
        engine = ChatEngine()
        
        with patch('ollama.chat') as mock_ollama:
            # Simulate connection error
            mock_ollama.side_effect = Exception("Connection refused")
            
            messages = [{"sender": "user", "content": "Test question"}]
            
            result = await engine.generate_response(
                messages=messages,
                model="llama3:latest",
                rag_method="No Specific RAG Method"
            )
            
            # Should handle error gracefully
            assert "encountered an error" in result["result"]
            assert result["duration"] >= 0
    
    @pytest.mark.asyncio
    async def test_local_index_unavailable_fallback(self):
        """Test fallback when local index is unavailable"""
        engine = ChatEngine()
        
        with patch.object(engine, '_get_local_query_engine') as mock_get_query_engine:
            # Simulate index not available
            mock_get_query_engine.return_value = None
            
            messages = [{"sender": "user", "content": "Question about local documents"}]
            
            result = await engine.generate_response(
                messages=messages,
                model="llama3:latest",
                rag_method="Planning Workflow",
                retrieval_method="local context only",
                preset="missing_preset"
            )
            
            # Should provide appropriate fallback response
            assert isinstance(result["result"], str)
            assert result["duration"] >= 0
    
    @pytest.mark.asyncio
    async def test_web_search_api_failure_fallback(self):
        """Test fallback when web search API fails"""
        engine = ChatEngine()
        
        with patch('chat_engine.RAG_methods.GoogleCustomSearchTool') as mock_google_class, \
             patch.object(engine, '_get_local_query_engine') as mock_get_query_engine:
            
            # Mock local query engine
            mock_query_engine = Mock()
            mock_query_engine.query.return_value = Mock(response="Local fallback response")
            mock_get_query_engine.return_value = mock_query_engine
            
            # Mock Google search tool to fail
            mock_google_tool = Mock()
            mock_google_tool.search.side_effect = Exception("API quota exceeded")
            mock_google_class.return_value = mock_google_tool
            
            # Mock asyncio.to_thread for multi-strategy workflow
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.side_effect = [
                    Mock(response="Local response"),  # Local query succeeds
                    Exception("Web search failed"),   # Web query fails
                    Mock(response="Local response only")  # Synthesis with local only
                ]
                
                messages = [{"sender": "user", "content": "Question requiring web search"}]
                
                result = await engine.generate_response(
                    messages=messages,
                    model="llama3:latest",
                    rag_method="Multi-Strategy Workflow",
                    retrieval_method="Hybrid context",
                    preset="default"
                )
                
                # Should handle web failure gracefully and use local response
                assert isinstance(result["result"], str)
                assert result["duration"] >= 0
    
    def test_malformed_rag_configuration(self):
        """Test handling of malformed RAG configuration"""
        engine = ChatEngine()
        
        # Test with invalid configuration
        with pytest.raises(Exception):
            # This should be handled gracefully in a real implementation
            pass  # Placeholder - would test actual malformed config handling


class TestRAGAccuracy:
    """Test accuracy and correctness of RAG components"""
    
    def test_claim_extraction_accuracy(self, mock_llm):
        """Test accuracy of claim extraction"""
        # Text with clear factual claims
        test_text = """
        The Eiffel Tower is located in Paris, France and was completed in 1889.
        It stands 324 meters tall including its antennas.
        The tower was designed by Gustave Eiffel and serves as a global cultural icon.
        """
        
        # Mock realistic claim extraction
        mock_llm.complete.return_value = """
        CLAIM: The Eiffel Tower is located in Paris, France
        CLAIM: The Eiffel Tower was completed in 1889
        CLAIM: The Eiffel Tower stands 324 meters tall including its antennas
        CLAIM: The tower was designed by Gustave Eiffel
        CLAIM: The Eiffel Tower serves as a global cultural icon
        """
        
        extractor = FactualClaimExtractor(mock_llm)
        claims = extractor.extract_claims(test_text)
        
        # Verify claims were extracted accurately
        assert len(claims) == 5
        assert any("Paris, France" in claim for claim in claims)
        assert any("1889" in claim for claim in claims)
        assert any("324 meters" in claim for claim in claims)
        assert any("Gustave Eiffel" in claim for claim in claims)
        assert any("cultural icon" in claim for claim in claims)
    
    @patch('time.perf_counter')
    def test_verification_accuracy_local_priority(self, mock_time, mock_llm, mock_local_query_engine, mock_google_search_tool):
        """Test verification accuracy with local evidence priority"""
        mock_time.return_value = 1.0
        
        # Setup local evidence (should be prioritized)
        mock_local_query_engine.query.return_value = Mock()
        mock_local_query_engine.query.return_value.__str__ = lambda x: "According to our documentation, anthropometry is the scientific study of human body measurements and proportions, used extensively in garment sizing and fit optimization."
        
        # Setup conflicting web evidence (should be deprioritized)
        mock_google_search_tool.search.return_value = "Anthropometry is primarily used in archaeology and forensics."
        
        # Mock LLM analysis that correctly prioritizes local evidence
        mock_llm.complete.return_value = """
        VERDICT: SUPPORTED
        CONFIDENCE: 0.95
        CORRECTION: None
        REASONING: Local document evidence provides authoritative definition for this specialized context
        """
        
        verifier = FactVerifier(mock_llm, mock_local_query_engine, mock_google_search_tool)
        result = verifier.verify_claim("Anthropometry is used for garment sizing", use_local=True, use_web=True)
        
        # Should prioritize local evidence
        assert result['is_supported'] == True
        assert result['confidence'] >= 0.95
        assert len(result['evidence']) == 1  # Only local evidence used
        assert result['evidence'][0]['source'] == 'local_knowledge'


if __name__ == "__main__":
    pytest.main([__file__])