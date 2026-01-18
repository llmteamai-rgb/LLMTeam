"""Tests for Escalation in GroupOrchestrator."""

import pytest
from datetime import datetime

from llmteam.roles.group_orch import (
    GroupOrchestrator,
    Escalation,
    EscalationLevel,
    EscalationAction,
    EscalationDecision,
    PipelineStatus,
)


class TestEscalation:
    """Tests for Escalation dataclass."""

    def test_create_escalation(self):
        """Test creating an escalation."""
        escalation = Escalation(
            escalation_id="esc-123",
            level=EscalationLevel.WARNING,
            source_pipeline="analysis_pipeline",
            reason="Model confidence below threshold",
            context={"confidence": 0.3},
        )

        assert escalation.escalation_id == "esc-123"
        assert escalation.level == EscalationLevel.WARNING
        assert escalation.source_pipeline == "analysis_pipeline"
        assert escalation.reason == "Model confidence below threshold"
        assert escalation.context["confidence"] == 0.3

    def test_escalation_to_dict(self):
        """Test escalation serialization."""
        ts = datetime(2024, 1, 15, 12, 0, 0)
        escalation = Escalation(
            escalation_id="esc-123",
            level=EscalationLevel.CRITICAL,
            source_pipeline="pipeline_1",
            reason="Error occurred",
            timestamp=ts,
        )

        data = escalation.to_dict()

        assert data["escalation_id"] == "esc-123"
        assert data["level"] == "critical"
        assert data["source_pipeline"] == "pipeline_1"
        assert data["timestamp"] == ts.isoformat()


class TestEscalationDecision:
    """Tests for EscalationDecision."""

    def test_create_decision(self):
        """Test creating an escalation decision."""
        decision = EscalationDecision(
            action=EscalationAction.REDIRECT,
            target_pipeline="backup_pipeline",
            message="Redirecting due to error",
        )

        assert decision.action == EscalationAction.REDIRECT
        assert decision.target_pipeline == "backup_pipeline"
        assert decision.message == "Redirecting due to error"

    def test_decision_to_dict(self):
        """Test decision serialization."""
        decision = EscalationDecision(
            action=EscalationAction.HUMAN_REVIEW,
            message="Needs review",
            metadata={"severity": "high"},
        )

        data = decision.to_dict()

        assert data["action"] == "human_review"
        assert data["message"] == "Needs review"
        assert data["metadata"]["severity"] == "high"


class TestGroupOrchestratorEscalation:
    """Tests for GroupOrchestrator escalation handling."""

    @pytest.fixture
    def group(self):
        """Create a GroupOrchestrator."""
        return GroupOrchestrator(group_id="test_group")

    @pytest.fixture
    def group_with_pipelines(self, group):
        """Create a GroupOrchestrator with mock pipelines."""
        # Register mock pipelines
        class MockPipeline:
            def __init__(self, pid):
                self.pipeline_id = pid

        group.register_pipeline(MockPipeline("pipeline_1"))
        group.register_pipeline(MockPipeline("pipeline_2"))
        return group

    async def test_handle_escalation_info(self, group):
        """Test handling INFO level escalation."""
        escalation = Escalation(
            escalation_id="esc-1",
            level=EscalationLevel.INFO,
            source_pipeline="pipeline_1",
            reason="Informational message",
        )

        decision = await group.handle_escalation(escalation)

        assert decision.action == EscalationAction.ACKNOWLEDGE

    async def test_handle_escalation_warning_redirect(self, group_with_pipelines):
        """Test handling WARNING level escalation with redirect."""
        # Set pipeline_1 as running, pipeline_2 as idle
        group_with_pipelines._statuses["pipeline_1"].status = "running"
        group_with_pipelines._statuses["pipeline_2"].status = "idle"

        escalation = Escalation(
            escalation_id="esc-2",
            level=EscalationLevel.WARNING,
            source_pipeline="pipeline_1",
            reason="Resource constraint",
        )

        decision = await group_with_pipelines.handle_escalation(escalation)

        assert decision.action == EscalationAction.REDIRECT
        assert decision.target_pipeline == "pipeline_2"

    async def test_handle_escalation_warning_retry(self, group):
        """Test handling WARNING level escalation with retry when no alternative available."""
        escalation = Escalation(
            escalation_id="esc-3",
            level=EscalationLevel.WARNING,
            source_pipeline="pipeline_1",
            reason="Temporary failure",
        )

        decision = await group.handle_escalation(escalation)

        assert decision.action == EscalationAction.RETRY

    async def test_handle_escalation_critical(self, group):
        """Test handling CRITICAL level escalation."""
        escalation = Escalation(
            escalation_id="esc-4",
            level=EscalationLevel.CRITICAL,
            source_pipeline="pipeline_1",
            reason="Data integrity issue",
        )

        decision = await group.handle_escalation(escalation)

        assert decision.action == EscalationAction.HUMAN_REVIEW
        assert "escalation" in decision.metadata

    async def test_handle_escalation_emergency(self, group):
        """Test handling EMERGENCY level escalation."""
        escalation = Escalation(
            escalation_id="esc-5",
            level=EscalationLevel.EMERGENCY,
            source_pipeline="pipeline_1",
            reason="System failure",
        )

        decision = await group.handle_escalation(escalation)

        assert decision.action == EscalationAction.ABORT

    async def test_custom_escalation_handler(self, group):
        """Test using custom escalation handler."""
        escalation = Escalation(
            escalation_id="esc-6",
            level=EscalationLevel.WARNING,
            source_pipeline="pipeline_1",
            reason="Custom handling needed",
        )

        def custom_handler(esc):
            return EscalationDecision(
                action=EscalationAction.ACKNOWLEDGE,
                message="Handled by custom handler",
            )

        decision = await group.handle_escalation(escalation, handler=custom_handler)

        assert decision.action == EscalationAction.ACKNOWLEDGE
        assert "custom handler" in decision.message

    async def test_escalation_history_tracking(self, group):
        """Test that escalations are tracked in history."""
        escalation1 = Escalation(
            escalation_id="esc-1",
            level=EscalationLevel.INFO,
            source_pipeline="pipeline_1",
            reason="Info 1",
        )
        escalation2 = Escalation(
            escalation_id="esc-2",
            level=EscalationLevel.WARNING,
            source_pipeline="pipeline_2",
            reason="Warning 1",
        )

        await group.handle_escalation(escalation1)
        await group.handle_escalation(escalation2)

        history = group.get_escalation_history()

        assert len(history) == 2
        # Most recent first
        assert history[0].escalation_id == "esc-2"
        assert history[1].escalation_id == "esc-1"

    async def test_escalation_history_filtered_by_pipeline(self, group):
        """Test filtering escalation history by pipeline."""
        await group.handle_escalation(Escalation(
            escalation_id="esc-1",
            level=EscalationLevel.INFO,
            source_pipeline="pipeline_1",
            reason="From pipeline 1",
        ))
        await group.handle_escalation(Escalation(
            escalation_id="esc-2",
            level=EscalationLevel.INFO,
            source_pipeline="pipeline_2",
            reason="From pipeline 2",
        ))

        history = group.get_escalation_history(pipeline_id="pipeline_1")

        assert len(history) == 1
        assert history[0].source_pipeline == "pipeline_1"

    async def test_escalation_history_filtered_by_level(self, group):
        """Test filtering escalation history by level."""
        await group.handle_escalation(Escalation(
            escalation_id="esc-1",
            level=EscalationLevel.INFO,
            source_pipeline="pipeline_1",
            reason="Info",
        ))
        await group.handle_escalation(Escalation(
            escalation_id="esc-2",
            level=EscalationLevel.WARNING,
            source_pipeline="pipeline_1",
            reason="Warning",
        ))

        history = group.get_escalation_history(level=EscalationLevel.WARNING)

        assert len(history) == 1
        assert history[0].level == EscalationLevel.WARNING


class TestGroupOrchestratorMetrics:
    """Tests for GroupOrchestrator metrics collection."""

    @pytest.fixture
    def group(self):
        return GroupOrchestrator(group_id="test_group")

    async def test_collect_metrics_basic(self, group):
        """Test basic metrics collection."""
        metrics = group.collect_metrics()

        assert "total_pipelines" in metrics
        assert "escalations" in metrics
        assert "health_score" in metrics
        assert "collected_at" in metrics

    async def test_collect_metrics_with_escalations(self, group):
        """Test metrics collection includes escalation stats."""
        await group.handle_escalation(Escalation(
            escalation_id="esc-1",
            level=EscalationLevel.WARNING,
            source_pipeline="pipeline_1",
            reason="Warning",
        ))
        await group.handle_escalation(Escalation(
            escalation_id="esc-2",
            level=EscalationLevel.CRITICAL,
            source_pipeline="pipeline_1",
            reason="Critical",
        ))

        metrics = group.collect_metrics()

        assert metrics["escalations"]["total"] == 2
        assert metrics["escalations"]["by_level"]["warning"] == 1
        assert metrics["escalations"]["by_level"]["critical"] == 1
        assert metrics["escalations"]["by_pipeline"]["pipeline_1"] == 2

    async def test_health_score_decreases_with_critical(self, group):
        """Test health score decreases with critical escalations."""
        # Get baseline
        baseline_metrics = group.collect_metrics()
        baseline_health = baseline_metrics["health_score"]

        # Add critical escalations
        for i in range(3):
            await group.handle_escalation(Escalation(
                escalation_id=f"esc-{i}",
                level=EscalationLevel.CRITICAL,
                source_pipeline="pipeline_1",
                reason="Critical issue",
            ))

        metrics = group.collect_metrics()

        assert metrics["health_score"] < baseline_health
