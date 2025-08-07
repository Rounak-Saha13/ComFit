"""
Test configuration and fixtures for RAG methods and retrieval testing
"""
import pytest
import os
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# Test environment setup
@pytest.fixture(scope="session")
def test_env():
    """Setup test environment variables"""
    # Mock environment variables for testing
    test_vars = {
        "GOOGLE_API_KEY": "test_api_key",
        "GOOGLE_CSE_ID": "test_cse_id",
        "OLLAMA_BASE_URL": "localhost:11434"
    }
    
    # Store original values
    original_vars = {}
    for key, value in test_vars.items():
        original_vars[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_vars
    
    # Restore original values
    for key, original_value in original_vars.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = Mock()
    llm.complete.return_value = Mock(response="Test LLM response")
    return llm

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing"""
    return Mock()

@pytest.fixture
def mock_local_query_engine():
    """Mock local query engine for testing"""
    engine = Mock()
    engine.query.return_value = Mock(response="Test local query response")
    return engine

@pytest.fixture
def mock_google_search_tool():
    """Mock Google search tool for testing"""
    tool = Mock()
    tool.search.return_value = """
    Result 1: Title: Test Result 1
    Snippet: This is a test search result
    Link: https://example.com/1
    ---
    Result 2: Title: Test Result 2
    Snippet: This is another test search result
    Link: https://example.com/2
    ---
    """
    return tool

@pytest.fixture
def sample_text_with_claims():
    """Sample text containing factual claims for testing"""
    return """
    The human heart has four chambers: two atria and two ventricles.
    Water boils at 100 degrees Celsius at sea level.
    The capital of France is Paris.
    Python is a programming language that was first released in 1991.
    The speed of light in vacuum is approximately 299,792,458 meters per second.
    """

@pytest.fixture
def sample_claims():
    """Sample factual claims for testing"""
    return [
        "The human heart has four chambers",
        "Water boils at 100 degrees Celsius at sea level",
        "The capital of France is Paris",
        "Python was first released in 1991",
        "The speed of light in vacuum is approximately 299,792,458 meters per second"
    ]

@pytest.fixture
def sample_context_memory():
    """Sample context memory for MCP testing"""
    return {
        "conversation_history": [
            {"sender": "user", "content": "Hello"},
            {"sender": "ai", "content": "Hi there!"}
        ],
        "system_prompt": "You are a helpful assistant",
        "model_config": {
            "model": "llama3:latest",
            "temperature": 0.7,
            "top_p": 0.9,
            "rag_method": "RAC Enhanced Hybrid RAG",
            "retrieval_method": "Hybrid context",
            "preset": "default"
        }
    }

@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Create a minimal PDF-like file (not a real PDF, but for testing purposes)
        f.write(b"%PDF-1.4\n")
        f.write(b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n")
        f.write(b"trailer\n<</Size 1/Root 1 0 R>>\n")
        f.write(b"%%EOF\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass

@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
        f.write("""
        This is a test document for RAG testing.
        
        The document contains information about comfort and fitting.
        Anthropometry is the study of human body measurements.
        Product fit is crucial for customer satisfaction.
        Sizing charts help determine the right fit for clothing.
        
        This text file simulates content that would be indexed for retrieval.
        """)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass

@pytest.fixture
def mock_vector_index():
    """Mock vector store index for testing"""
    index = Mock()
    query_engine = Mock()
    query_engine.query.return_value = Mock(response="Mocked index response")
    index.as_query_engine.return_value = query_engine
    return index

@pytest.fixture
def mock_react_agent():
    """Mock ReAct agent for testing"""
    agent = Mock()
    agent.chat.return_value = Mock(response="Agent response for testing")
    return agent

# Async test helper
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Mock data fixtures
@pytest.fixture
def mock_verification_result():
    """Mock verification result structure"""
    return {
        'claim': 'Test claim',
        'is_supported': True,
        'confidence': 0.8,
        'evidence': [
            {
                'source': 'local_knowledge',
                'content': 'Supporting evidence from local documents',
                'confidence': 0.9,
                'query_used': 'test query'
            }
        ],
        'correction_suggestion': None,
        'warning': None
    }

@pytest.fixture
def mock_rac_result():
    """Mock RAC correction result"""
    return {
        'original_response': 'Original response text',
        'corrected_response': 'Corrected response text',
        'claims_analyzed': 3,
        'corrections_made': 1,
        'verification_results': [],
        'corrections_applied': [
            {
                'original_claim': 'Original incorrect claim',
                'correction': 'Corrected claim',
                'confidence': 0.85,
                'evidence_sources': ['local_knowledge']
            }
        ],
        'uncertain_claims': [],
        'average_confidence': 0.75
    }

@pytest.fixture
def mock_mcp_response():
    """Mock MCP response structure"""
    return {
        "final_answer": "Test final answer",
        "trace": ["Step 1: Processing", "Step 2: Verification", "Step 3: Response"],
        "confidence_score": 0.8
    }