"""
Tests for ChatEngine RAG method implementations
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os

# Import the ChatEngine class
from chat_engine.client import ChatEngine


class TestChatEngine:
    """Test the ChatEngine class and its RAG method implementations"""
    
    def test_init(self):
        """Test ChatEngine initialization"""
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "test:11434"}):
            engine = ChatEngine()
            assert engine.ollama_base_url == "test:11434"
            assert engine.default_model == "llama3:latest"
    
    def test_init_default_url(self):
        """Test ChatEngine initialization with default URL"""
        with patch.dict(os.environ, {}, clear=True):
            engine = ChatEngine()
            assert engine.ollama_base_url == "192.168.0.240:11434"
    
    @patch('ollama.show')
    def test_validate_model_success(self, mock_show):
        """Test successful model validation"""
        mock_show.return_value = {"model": "test_model"}
        
        engine = ChatEngine()
        result = engine.validate_model("test_model")
        
        assert result == True
        mock_show.assert_called_once_with("test_model")
    
    @patch('ollama.show')
    def test_validate_model_failure(self, mock_show):
        """Test model validation failure"""
        mock_show.side_effect = Exception("Model not found")
        
        engine = ChatEngine()
        result = engine.validate_model("invalid_model")
        
        assert result == False
    
    def test_get_local_query_engine_from_document_manager(self):
        """Test getting query engine from document manager"""
        engine = ChatEngine()
        
        # Mock document manager to return an index
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_index.as_query_engine.return_value = mock_query_engine
        engine.document_manager.load_index.return_value = mock_index
        
        with patch('llama_index.llms.ollama.Ollama') as mock_ollama:
            result = engine._get_local_query_engine("test_preset")
            
            assert result == mock_query_engine
            engine.document_manager.load_index.assert_called_once_with("test_preset")
            mock_index.as_query_engine.assert_called_once()
    
    @patch('llama_index.core.load_index_from_storage')
    @patch('llama_index.core.StorageContext.from_defaults')
    @patch('pathlib.Path.exists')
    def test_get_local_query_engine_from_root_store(self, mock_exists, mock_storage_context, mock_load_index):
        """Test getting query engine from root vector store"""
        engine = ChatEngine()
        
        # Mock document manager to return None
        engine.document_manager.load_index.return_value = None
        
        # Mock root vector store exists
        mock_exists.return_value = True
        
        # Mock successful index loading
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_load_index.return_value = mock_index
        
        with patch('llama_index.llms.ollama.Ollama') as mock_ollama:
            result = engine._get_local_query_engine("test_preset")
            
            assert result == mock_query_engine
    
    def test_get_local_query_engine_not_found(self):
        """Test getting query engine when no index is found"""
        engine = ChatEngine()
        
        # Mock document manager to return None
        engine.document_manager.load_index.return_value = None
        
        with patch('pathlib.Path.exists', return_value=False):
            result = engine._get_local_query_engine("test_preset")
            
            assert result is None


class TestChatEngineGenerateResponse:
    """Test ChatEngine response generation methods"""
    
    @pytest.mark.asyncio
    async def test_generate_response_default_method(self):
        """Test response generation with default method"""
        engine = ChatEngine()
        
        messages = [
            {"sender": "user", "content": "Hello"},
            {"sender": "user", "content": "How are you?"}
        ]
        
        with patch.object(engine, '_process_default') as mock_process:
            mock_process.return_value = "Default response"
            
            result = await engine.generate_response(
                messages=messages,
                model="test_model",
                rag_method="No Specific RAG Method"
            )
            
            assert result["result"] == "Default response"
            assert "duration" in result
            assert "ai_message" in result
            assert result["ai_message"]["content"] == "Default response"
    
    @pytest.mark.asyncio
    async def test_generate_response_planning_workflow(self):
        """Test response generation with planning workflow"""
        engine = ChatEngine()
        
        messages = [{"sender": "user", "content": "Test question"}]
        
        with patch.object(engine, '_process_planning_workflow') as mock_process:
            mock_process.return_value = "Planning workflow response"
            
            result = await engine.generate_response(
                messages=messages,
                model="test_model",
                rag_method="Planning Workflow"
            )
            
            assert result["result"] == "Planning workflow response"
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_multi_step_query(self):
        """Test response generation with multi-step query engine"""
        engine = ChatEngine()
        
        messages = [{"sender": "user", "content": "Test question"}]
        
        with patch.object(engine, '_process_multi_step_query') as mock_process:
            mock_process.return_value = "Multi-step response"
            
            result = await engine.generate_response(
                messages=messages,
                model="test_model",
                rag_method="Multi-Step Query Engine"
            )
            
            assert result["result"] == "Multi-step response"
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_multi_strategy_workflow(self):
        """Test response generation with multi-strategy workflow"""
        engine = ChatEngine()
        
        messages = [{"sender": "user", "content": "Test question"}]
        
        with patch.object(engine, '_process_multi_strategy_workflow') as mock_process:
            mock_process.return_value = "Multi-strategy response"
            
            result = await engine.generate_response(
                messages=messages,
                model="test_model",
                rag_method="Multi-Strategy Workflow"
            )
            
            assert result["result"] == "Multi-strategy response"
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_rac_enhanced_rag(self):
        """Test response generation with RAC Enhanced Hybrid RAG"""
        engine = ChatEngine()
        
        messages = [{"sender": "user", "content": "Test question"}]
        
        with patch.object(engine, '_process_rac_enhanced_rag') as mock_process:
            mock_process.return_value = "RAC enhanced response"
            
            result = await engine.generate_response(
                messages=messages,
                model="test_model",
                rag_method="RAC Enhanced Hybrid RAG"
            )
            
            assert result["result"] == "RAC enhanced response"
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_exception_handling(self):
        """Test response generation exception handling"""
        engine = ChatEngine()
        
        messages = [{"sender": "user", "content": "Test question"}]
        
        with patch.object(engine, '_process_default') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            result = await engine.generate_response(
                messages=messages,
                model="test_model"
            )
            
            assert "Error generating response" in result["result"]
            assert "Test error" in result["result"]


class TestChatEngineDefaultProcessing:
    """Test ChatEngine default processing method"""
    
    @pytest.mark.asyncio
    @patch('ollama.chat')
    @patch('ollama.list')
    async def test_process_default_success(self, mock_list, mock_chat):
        """Test successful default processing"""
        engine = ChatEngine()
        
        # Mock Ollama responses
        mock_list.return_value = {"models": []}
        mock_chat.return_value = {
            "message": {
                "content": "Test response from Ollama"
            }
        }
        
        context_memory = {
            "model_config": {
                "model": "test_model",
                "temperature": 0.7
            },
            "system_prompt": "Test system prompt",
            "conversation_history": [
                {"sender": "user", "content": "Previous message"}
            ]
        }
        
        result = await engine._process_default("Test user message", context_memory)
        
        assert result == "Test response from Ollama"
        mock_chat.assert_called_once()
        
        # Verify call arguments
        call_args = mock_chat.call_args
        assert call_args[1]["model"] == "test_model"
        assert len(call_args[1]["messages"]) == 3  # system + history + current
        assert call_args[1]["options"]["temperature"] == 0.7
    
    @pytest.mark.asyncio
    @patch('ollama.chat')
    async def test_process_default_empty_response(self, mock_chat):
        """Test default processing with empty Ollama response"""
        engine = ChatEngine()
        
        # Mock empty Ollama response
        mock_chat.return_value = {"message": {"content": ""}}
        
        context_memory = {
            "model_config": {"model": "test_model"},
            "system_prompt": "Test prompt",
            "conversation_history": []
        }
        
        result = await engine._process_default("Test message", context_memory)
        
        assert "couldn't generate a response" in result
        assert "Test message" in result
    
    @pytest.mark.asyncio
    @patch('ollama.chat')
    async def test_process_default_ollama_error(self, mock_chat):
        """Test default processing with Ollama error"""
        engine = ChatEngine()
        
        # Mock Ollama error
        mock_chat.side_effect = Exception("Ollama connection error")
        
        context_memory = {
            "model_config": {"model": "test_model"},
            "system_prompt": "Test prompt",
            "conversation_history": []
        }
        
        result = await engine._process_default("Test message", context_memory)
        
        assert "encountered an error" in result
        assert "Test message" in result


class TestChatEnginePlanningWorkflow:
    """Test ChatEngine planning workflow processing"""
    
    @pytest.mark.asyncio
    @patch('chat_engine.RAG_methods.run_planning_workflow')
    async def test_process_planning_workflow_success(self, mock_run_planning):
        """Test successful planning workflow processing"""
        engine = ChatEngine()
        
        # Mock successful workflow execution
        mock_run_planning.return_value = "Planning workflow result"
        
        # Mock query engine
        mock_query_engine = Mock()
        with patch.object(engine, '_get_local_query_engine', return_value=mock_query_engine):
            context_memory = {
                "model_config": {
                    "model": "test_model",
                    "preset": "test_preset",
                    "retrieval_method": "local context only"
                }
            }
            
            result = await engine._process_planning_workflow("test query", context_memory)
            
            assert result == "Planning workflow result"
            mock_run_planning.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test_key", "GOOGLE_CSE_ID": "test_id"})
    @patch('chat_engine.RAG_methods.run_planning_workflow')
    @patch('chat_engine.RAG_methods.GoogleCustomSearchTool')
    async def test_process_planning_workflow_with_web_search(self, mock_google_tool, mock_run_planning):
        """Test planning workflow with web search enabled"""
        engine = ChatEngine()
        
        mock_run_planning.return_value = "Planning with web search result"
        mock_google_instance = Mock()
        mock_google_tool.return_value = mock_google_instance
        
        context_memory = {
            "model_config": {
                "model": "test_model",
                "preset": "test_preset",
                "retrieval_method": "Hybrid context"
            }
        }
        
        with patch.object(engine, '_get_local_query_engine', return_value=None):
            result = await engine._process_planning_workflow("test query", context_memory)
            
            assert result == "Planning with web search result"
            mock_google_tool.assert_called_once_with(
                api_key="test_key",
                cse_id="test_id",
                num_results=3
            )
    
    @pytest.mark.asyncio
    async def test_process_planning_workflow_no_tools(self):
        """Test planning workflow when no tools are available"""
        engine = ChatEngine()
        
        with patch.object(engine, '_get_local_query_engine', return_value=None):
            with patch('chat_engine.RAG_methods.run_planning_workflow') as mock_run_planning:
                mock_run_planning.return_value = "Fallback response"
                
                context_memory = {
                    "model_config": {
                        "model": "test_model",
                        "preset": "test_preset",
                        "retrieval_method": "local context only"
                    }
                }
                
                result = await engine._process_planning_workflow("test query", context_memory)
                
                assert result == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_process_planning_workflow_error(self):
        """Test planning workflow error handling"""
        engine = ChatEngine()
        
        with patch.object(engine, '_get_local_query_engine', side_effect=Exception("Test error")):
            context_memory = {
                "model_config": {
                    "model": "test_model",
                    "preset": "test_preset",
                    "retrieval_method": "local context only"
                }
            }
            
            result = await engine._process_planning_workflow("test query", context_memory)
            
            assert "Planning workflow encountered an error" in result
            assert "Test error" in result


class TestChatEngineMultiStepQuery:
    """Test ChatEngine multi-step query engine processing"""
    
    @pytest.mark.asyncio
    @patch('chat_engine.RAG_methods.run_multi_step_query_engine_workflow')
    async def test_process_multi_step_query_success(self, mock_run_multi_step):
        """Test successful multi-step query processing"""
        engine = ChatEngine()
        
        mock_run_multi_step.return_value = "Multi-step result"
        
        mock_query_engine = Mock()
        with patch.object(engine, '_get_local_query_engine', return_value=mock_query_engine):
            context_memory = {
                "model_config": {
                    "model": "test_model",
                    "preset": "test_preset",
                    "retrieval_method": "local context only"
                }
            }
            
            result = await engine._process_multi_step_query("test query", context_memory)
            
            assert result == "Multi-step result"
            mock_run_multi_step.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_multi_step_query_no_local_engine(self):
        """Test multi-step query when local engine is not available"""
        engine = ChatEngine()
        
        with patch.object(engine, '_get_local_query_engine', return_value=None):
            context_memory = {
                "model_config": {
                    "model": "test_model",
                    "preset": "test_preset",
                    "retrieval_method": "local context only"
                }
            }
            
            result = await engine._process_multi_step_query("test query", context_memory)
            
            assert "Local document search not available" in result
            assert "test_preset" in result


class TestChatEngineRAC:
    """Test ChatEngine RAC Enhanced Hybrid RAG processing"""
    
    @pytest.mark.asyncio
    @patch('chat_engine.RAG_methods.process_model_context_query')
    async def test_process_rac_enhanced_rag_success(self, mock_process_mcp):
        """Test successful RAC enhanced RAG processing"""
        engine = ChatEngine()
        
        mock_process_mcp.return_value = {
            "final_answer": "RAC enhanced answer",
            "trace": ["Step 1", "Step 2"],
            "confidence_score": 0.85
        }
        
        mock_query_engine = Mock()
        with patch.object(engine, '_get_local_query_engine', return_value=mock_query_engine):
            context_memory = {
                "model_config": {
                    "model": "test_model",
                    "preset": "test_preset",
                    "retrieval_method": "Hybrid context"
                }
            }
            
            result = await engine._process_rac_enhanced_rag("test query", context_memory)
            
            assert result == "RAC enhanced answer"
            mock_process_mcp.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_rac_enhanced_rag_no_local_engine(self):
        """Test RAC enhanced RAG when local engine is not available"""
        engine = ChatEngine()
        
        with patch.object(engine, '_get_local_query_engine', return_value=None):
            context_memory = {
                "model_config": {
                    "model": "test_model",
                    "preset": "test_preset",
                    "retrieval_method": "local context only"
                }
            }
            
            result = await engine._process_rac_enhanced_rag("test query", context_memory)
            
            assert "Local document search not available" in result


class TestChatEngineUtilities:
    """Test ChatEngine utility methods"""
    
    def test_list_available_vector_stores(self):
        """Test listing available vector stores"""
        engine = ChatEngine()
        
        # Mock document manager
        engine.document_manager.list_indexes.return_value = ["index1", "index2"]
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.glob', return_value=["file1.json", "file2.json"]):
                result = engine.list_available_vector_stores()
                
                assert "document_manager" in result
                assert "root_directory" in result
                assert result["document_manager"] == ["index1", "index2"]
                assert result["root_directory"] == ["default"]
    
    def test_list_available_vector_stores_no_root(self):
        """Test listing vector stores when root directory doesn't exist"""
        engine = ChatEngine()
        
        engine.document_manager.list_indexes.return_value = ["index1"]
        
        with patch('pathlib.Path.exists', return_value=False):
            result = engine.list_available_vector_stores()
            
            assert result["document_manager"] == ["index1"]
            assert result["root_directory"] == []
    
    def test_get_available_presets(self):
        """Test getting available presets"""
        engine = ChatEngine()
        
        with patch.object(engine, 'list_available_vector_stores') as mock_list:
            mock_list.return_value = {
                "document_manager": ["preset1", "preset2"],
                "root_directory": ["default", "preset1"]  # preset1 is duplicate
            }
            
            presets = engine.get_available_presets()
            
            # Should remove duplicates
            assert len(presets) == 3
            assert "preset1" in presets
            assert "preset2" in presets
            assert "default" in presets
    
    @pytest.mark.asyncio
    async def test_regenerate_response(self):
        """Test response regeneration"""
        engine = ChatEngine()
        
        messages = [
            {"id": "msg1", "sender": "user", "content": "Hello"},
            {"id": "msg2", "sender": "ai", "content": "Hi there"},
            {"id": "msg3", "sender": "user", "content": "How are you?"}
        ]
        
        with patch.object(engine, 'generate_response') as mock_generate:
            mock_generate.return_value = {"result": "Regenerated response"}
            
            result = await engine.regenerate_response(
                message_id="msg2",
                conversation_id="conv1",
                messages=messages,
                model="test_model"
            )
            
            # Should call generate_response with messages excluding the regenerated one
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[1]
            filtered_messages = call_args["messages"]
            
            # Should exclude message with id "msg2"
            assert len(filtered_messages) == 2
            assert all(msg["id"] != "msg2" for msg in filtered_messages)


if __name__ == "__main__":
    pytest.main([__file__])