# RAG Methods and Retrieval Strategies Test Suite

This comprehensive test suite validates the functionality, performance, and reliability of the RAG (Retrieval-Augmented Generation) methods and retrieval strategies implemented in the ComFit application.

## 📋 Test Coverage

### Core Components Tested

1. **FactualClaimExtractor** (`test_rag_methods.py`)

   - Claim extraction from LLM responses
   - Filtering of meta-information
   - Error handling and edge cases

2. **FactVerifier** (`test_rag_methods.py`)

   - Local and web evidence verification
   - Confidence scoring and thresholds
   - Reversal protection logic
   - Caching mechanisms

3. **RACCorrector** (`test_rag_methods.py`)

   - End-to-end correction pipeline
   - Testing mode functionality
   - Confidence cascade logic
   - Verification mode switching

4. **GoogleCustomSearchTool** (`test_rag_methods.py`)

   - Web search functionality
   - API error handling
   - Result formatting

5. **RAG Strategies** (`test_rag_methods.py`)

   - Planning Workflow
   - Multi-Step Query Engine
   - Multi-Strategy Workflow
   - Model Context Protocol processing

6. **ChatEngine** (`test_chat_engine.py`)

   - RAG method implementations
   - Query engine management
   - Error handling and fallbacks
   - Response generation pipeline

7. **Hybrid Bot & MCP** (`test_hybrid_bot.py`)

   - MCP schema validation
   - Enhanced claim extraction
   - Confidence cascade implementation
   - Integration testing

8. **Integration Tests** (`test_integration.py`)
   - End-to-end RAG pipelines
   - Performance benchmarks
   - Error resilience
   - Concurrent request handling

## 🚀 Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r tests/test_requirements.txt

# Run all tests
python tests/run_tests.py

# Run with coverage report
python tests/run_tests.py --coverage
```

### Test Categories

```bash
# Unit tests only
python tests/run_tests.py --unit

# Integration tests only
python tests/run_tests.py --integration

# Performance tests only
python tests/run_tests.py --performance

# Skip slow tests
python tests/run_tests.py --fast

# Run tests in parallel
python tests/run_tests.py --parallel 4
```

### Specific Test Execution

```bash
# Run specific test file
python tests/run_tests.py --file test_rag_methods.py

# Run specific test function
python tests/run_tests.py --file test_rag_methods.py --function test_extract_claims_success

# Run with verbose output
python tests/run_tests.py --verbose
```

### Alternative: Direct pytest

```bash
# Using pytest directly
pytest tests/ -v --cov=chat_engine

# With specific markers
pytest tests/ -m "unit and rag"

# With HTML report
pytest tests/ --html=reports/report.html
```

## 📊 Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.integration`: Integration tests for component interactions
- `@pytest.mark.performance`: Performance and load testing
- `@pytest.mark.slow`: Tests that take more than 5 seconds
- `@pytest.mark.rag`: Tests specifically for RAG functionality
- `@pytest.mark.retrieval`: Tests for retrieval strategies
- `@pytest.mark.correction`: Tests for RAC correction functionality
- `@pytest.mark.mcp`: Tests for Model Context Protocol
- `@pytest.mark.requires_llm`: Tests that require actual LLM connection
- `@pytest.mark.requires_web`: Tests that require web connectivity

## 🏗️ Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── pytest.ini                 # Pytest configuration
├── test_requirements.txt       # Test dependencies
├── run_tests.py               # Test runner script
├── test_rag_methods.py        # Core RAG component tests
├── test_chat_engine.py        # ChatEngine implementation tests
├── test_hybrid_bot.py         # Hybrid Bot and MCP tests
├── test_integration.py        # Integration and performance tests
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

For comprehensive testing, set these environment variables:

```bash
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_custom_search_engine_id"
export OLLAMA_BASE_URL="localhost:11434"
```

Note: Tests use mocked versions by default, so these are only needed for live integration tests.

### Test Configuration

Key configuration options in `pytest.ini`:

- **Coverage threshold**: 80% minimum
- **Test timeout**: 300 seconds
- **Async mode**: Auto-detection
- **Report generation**: HTML and XML formats

## 📈 Performance Testing

### Benchmarks Included

1. **Claim Extraction Performance**

   - Large text processing (10x multiplied content)
   - Response time under 5 seconds
   - Memory usage optimization

2. **Verification Caching**

   - Cache hit performance comparison
   - Memory efficiency validation
   - Concurrent access testing

3. **Concurrent Request Handling**
   - 10 parallel requests simulation
   - Response time consistency
   - Resource utilization monitoring

### Performance Thresholds

- **Claim extraction**: < 5 seconds for large texts
- **Verification caching**: 2nd call should be significantly faster than 1st
- **Concurrent handling**: 10 requests should complete within 10 seconds

## 🛠️ Mock Objects and Fixtures

### Key Fixtures (`conftest.py`)

- `mock_llm`: Mocked LLM instance
- `mock_local_query_engine`: Mocked local document search
- `mock_google_search_tool`: Mocked web search
- `sample_claims`: Predefined test claims
- `sample_context_memory`: MCP context for testing
- `temp_pdf_file` / `temp_text_file`: Temporary test files

### Mock Patterns

1. **LLM Responses**: Realistic formatted responses for different scenarios
2. **Query Engines**: Predictable responses for local document queries
3. **Web Search**: Formatted search results with proper structure
4. **Error Scenarios**: Controlled failure modes for resilience testing

## 🔍 Test Scenarios

### Unit Test Scenarios

1. **Normal Operation**: Standard functionality with expected inputs
2. **Edge Cases**: Empty inputs, malformed data, boundary conditions
3. **Error Handling**: LLM failures, network issues, missing dependencies
4. **Configuration Variants**: Different RAG methods, retrieval strategies, confidence thresholds

### Integration Test Scenarios

1. **Full Pipeline**: End-to-end RAG processing with all components
2. **Fallback Mechanisms**: Graceful degradation when components fail
3. **Performance Under Load**: Multiple concurrent requests
4. **Real-world Workflows**: Typical user interaction patterns

## 📋 Validation Criteria

### Functional Tests

- ✅ All RAG methods execute without errors
- ✅ Claim extraction identifies factual statements
- ✅ Fact verification produces confidence scores
- ✅ RAC correction applies when needed
- ✅ Confidence cascade works as expected
- ✅ Caching improves performance
- ✅ Error handling prevents crashes

### Performance Tests

- ✅ Response times within acceptable limits
- ✅ Memory usage remains stable
- ✅ Concurrent requests handled efficiently
- ✅ Cache hit rates improve over time

### Integration Tests

- ✅ Component interactions work correctly
- ✅ Data flows properly between modules
- ✅ Fallback mechanisms activate appropriately
- ✅ End-to-end workflows complete successfully

## 🐛 Debugging Test Failures

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Failures**: Check mock configuration matches actual interfaces
3. **Async Issues**: Verify proper async/await usage in tests
4. **Timeout Errors**: Increase timeout for slow tests or optimize code

### Debug Commands

```bash
# Run with maximum verbosity
python tests/run_tests.py --verbose

# Run single failing test
python tests/run_tests.py --file test_name.py --function test_function_name

# Generate detailed coverage report
python tests/run_tests.py --coverage --verbose
```

### Log Analysis

Test logs are configured to show:

- Component execution flow
- Performance timing information
- Error details with stack traces
- Mock interaction patterns

## 📚 Adding New Tests

### Test File Structure

```python
"""
Test module documentation
"""
import pytest
from unittest.mock import Mock, patch

class TestComponentName:
    """Test class for specific component"""

    def test_normal_operation(self, fixture_name):
        """Test normal operation"""
        # Arrange
        # Act
        # Assert

    @pytest.mark.slow
    def test_performance_scenario(self):
        """Test performance characteristics"""
        # Performance test logic

    @pytest.mark.integration
    async def test_integration_scenario(self):
        """Test integration with other components"""
        # Integration test logic
```

### Guidelines

1. **Naming**: Use descriptive test names that explain the scenario
2. **Markers**: Apply appropriate pytest markers for categorization
3. **Fixtures**: Use shared fixtures from `conftest.py` when possible
4. **Mocking**: Mock external dependencies to ensure test isolation
5. **Assertions**: Include multiple assertions to validate different aspects
6. **Documentation**: Add docstrings explaining test purpose and expected behavior

## 📄 Reports

After running tests, reports are generated in the `reports/` directory:

- **pytest_report.html**: Detailed test execution report
- **htmlcov/index.html**: Coverage report with line-by-line analysis
- **coverage.xml**: Machine-readable coverage data for CI/CD

## 🔄 Continuous Integration

### CI Configuration Example

```yaml
# .github/workflows/test-rag.yml
name: RAG Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"
      - name: Install dependencies
        run: |
          pip install -r backend/tests/test_requirements.txt
          pip install -r backend/requirements.txt
      - name: Run tests
        run: |
          cd backend
          python tests/run_tests.py --coverage --parallel 2
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: backend/reports/coverage.xml
```

This comprehensive test suite ensures the reliability, performance, and correctness of the RAG methods and retrieval strategies in the ComFit application.
