"""Tests for llmteam.agents.factory module."""

import pytest
from unittest.mock import MagicMock

from llmteam.agents.factory import AgentFactory
from llmteam.agents.types import AgentType
from llmteam.agents.config import LLMAgentConfig, RAGAgentConfig, KAGAgentConfig
from llmteam.agents.llm_agent import LLMAgent
from llmteam.agents.rag_agent import RAGAgent
from llmteam.agents.kag_agent import KAGAgent


class TestAgentFactory:
    """Tests for AgentFactory."""

    @pytest.fixture
    def mock_team(self):
        """Create mock LLMTeam."""
        team = MagicMock()
        team.team_id = "test_team"
        team._model = "gpt-4o-mini"
        return team

    def test_create_llm_agent_from_dict(self, mock_team):
        """Test creating LLM agent from dict config."""
        config = {
            "type": "llm",
            "role": "writer",
            "prompt": "Write about: {topic}",
        }
        agent = AgentFactory.create(team=mock_team, config=config)

        assert isinstance(agent, LLMAgent)
        assert agent.role == "writer"
        assert agent.agent_type == AgentType.LLM

    def test_create_rag_agent_from_dict(self, mock_team):
        """Test creating RAG agent from dict config."""
        config = {
            "type": "rag",
            "role": "retriever",
            "collection": "documents",
            "top_k": 10,
        }
        agent = AgentFactory.create(team=mock_team, config=config)

        assert isinstance(agent, RAGAgent)
        assert agent.role == "retriever"
        assert agent.agent_type == AgentType.RAG

    def test_create_kag_agent_from_dict(self, mock_team):
        """Test creating KAG agent from dict config."""
        config = {
            "type": "kag",
            "role": "knowledge",
            "max_hops": 3,
        }
        agent = AgentFactory.create(team=mock_team, config=config)

        assert isinstance(agent, KAGAgent)
        assert agent.role == "knowledge"
        assert agent.agent_type == AgentType.KAG

    def test_create_from_config_object(self, mock_team):
        """Test creating agent from AgentConfig object."""
        config = LLMAgentConfig(
            role="analyzer",
            prompt="Analyze: {text}",
            model="gpt-4o",
        )
        agent = AgentFactory.create(team=mock_team, config=config)

        assert isinstance(agent, LLMAgent)
        assert agent.role == "analyzer"

    def test_create_rag_from_config_object(self, mock_team):
        """Test creating RAG agent from RAGAgentConfig."""
        config = RAGAgentConfig(
            role="searcher",
            collection="docs",
            top_k=5,
        )
        agent = AgentFactory.create(team=mock_team, config=config)

        assert isinstance(agent, RAGAgent)
        assert agent.role == "searcher"

    def test_create_kag_from_config_object(self, mock_team):
        """Test creating KAG agent from KAGAgentConfig."""
        config = KAGAgentConfig(
            role="graph_query",
            max_hops=2,
        )
        agent = AgentFactory.create(team=mock_team, config=config)

        assert isinstance(agent, KAGAgent)
        assert agent.role == "graph_query"

    def test_invalid_type_raises_error(self, mock_team):
        """Test invalid agent type raises ValueError."""
        config = {
            "type": "invalid_type",
            "role": "test",
        }
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create(team=mock_team, config=config)

    def test_missing_type_defaults_to_llm(self, mock_team):
        """Test missing type defaults to LLM."""
        config = {"role": "test", "prompt": "test prompt"}
        agent = AgentFactory.create(team=mock_team, config=config)
        assert agent.agent_type == AgentType.LLM

    def test_missing_role_uses_default(self, mock_team):
        """Test missing role uses default from id or 'agent'."""
        config = {"type": "llm", "prompt": "test"}
        agent = AgentFactory.create(team=mock_team, config=config)
        # Should use default role
        assert agent.role == "agent"

    def test_team_required(self):
        """Test team parameter is required."""
        config = {"type": "llm", "role": "test", "prompt": "test"}
        with pytest.raises(TypeError):
            AgentFactory.create(team=None, config=config)

    def test_agent_id_defaults_to_role(self, mock_team):
        """Test agent_id defaults to role."""
        config = {
            "type": "llm",
            "role": "writer",
            "prompt": "Write: {input}",
        }
        agent = AgentFactory.create(team=mock_team, config=config)
        assert agent.agent_id == "writer"

    def test_custom_agent_id(self, mock_team):
        """Test custom agent_id."""
        config = {
            "type": "llm",
            "role": "writer",
            "id": "custom_writer_id",
            "prompt": "Write: {input}",
        }
        agent = AgentFactory.create(team=mock_team, config=config)
        assert agent.agent_id == "custom_writer_id"
        assert agent.role == "writer"

    def test_type_must_be_lowercase(self, mock_team):
        """Test type field must be lowercase."""
        # Uppercase should fail
        config = {
            "type": "LLM",
            "role": "test",
            "prompt": "test",
        }
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create(team=mock_team, config=config)

        # Lowercase should work
        config["type"] = "llm"
        agent = AgentFactory.create(team=mock_team, config=config)
        assert agent.agent_type == AgentType.LLM
