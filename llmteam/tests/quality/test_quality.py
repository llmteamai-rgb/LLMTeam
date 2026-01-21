"""
Tests for Quality Slider module (RFC-008).
"""

import pytest

from llmteam.quality import (
    QualityManager,
    QualityPreset,
    TaskComplexity,
    PipelineDepth,
    CostEstimate,
    CostEstimator,
)
from llmteam.quality.presets import (
    PRESET_VALUES,
    MODEL_MATRIX,
    GENERATION_PARAMS,
    get_quality_range,
)


class TestQualityPresets:
    """Tests for quality presets and constants."""

    def test_preset_values(self):
        """Preset values should be defined."""
        assert PRESET_VALUES["draft"] == 20
        assert PRESET_VALUES["economy"] == 30
        assert PRESET_VALUES["balanced"] == 50
        assert PRESET_VALUES["production"] == 75
        assert PRESET_VALUES["best"] == 95

    def test_quality_preset_enum(self):
        """QualityPreset enum should have all presets."""
        assert QualityPreset.DRAFT.value == "draft"
        assert QualityPreset.ECONOMY.value == "economy"
        assert QualityPreset.BALANCED.value == "balanced"
        assert QualityPreset.PRODUCTION.value == "production"
        assert QualityPreset.BEST.value == "best"
        assert QualityPreset.AUTO.value == "auto"

    def test_task_complexity_enum(self):
        """TaskComplexity enum should have all levels."""
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MEDIUM.value == "medium"
        assert TaskComplexity.COMPLEX.value == "complex"

    def test_pipeline_depth_enum(self):
        """PipelineDepth enum should have all depths."""
        assert PipelineDepth.SHALLOW.value == "shallow"
        assert PipelineDepth.MEDIUM.value == "medium"
        assert PipelineDepth.DEEP.value == "deep"

    def test_model_matrix_structure(self):
        """MODEL_MATRIX should have correct structure."""
        assert "simple" in MODEL_MATRIX
        assert "medium" in MODEL_MATRIX
        assert "complex" in MODEL_MATRIX

        for complexity in ["simple", "medium", "complex"]:
            matrix = MODEL_MATRIX[complexity]
            assert (0, 30) in matrix
            assert (30, 70) in matrix
            assert (70, 100) in matrix

    def test_generation_params_structure(self):
        """GENERATION_PARAMS should have correct structure."""
        assert (0, 30) in GENERATION_PARAMS
        assert (30, 70) in GENERATION_PARAMS
        assert (70, 100) in GENERATION_PARAMS

        for params in GENERATION_PARAMS.values():
            assert "max_tokens" in params
            assert "temperature" in params

    def test_get_quality_range(self):
        """get_quality_range should return correct ranges."""
        assert get_quality_range(0) == (0, 30)
        assert get_quality_range(15) == (0, 30)
        assert get_quality_range(29) == (0, 30)
        assert get_quality_range(30) == (30, 70)
        assert get_quality_range(50) == (30, 70)
        assert get_quality_range(69) == (30, 70)
        assert get_quality_range(70) == (70, 100)
        assert get_quality_range(85) == (70, 100)
        assert get_quality_range(100) == (70, 100)


class TestQualityManager:
    """Tests for QualityManager class."""

    def test_init_with_default(self):
        """Default quality should be 50."""
        manager = QualityManager()
        assert manager.quality == 50

    def test_init_with_int(self):
        """Init with integer quality."""
        manager = QualityManager(quality=70)
        assert manager.quality == 70

    def test_init_with_preset_string(self):
        """Init with preset string."""
        manager = QualityManager(quality="draft")
        assert manager.quality == 20

        manager = QualityManager(quality="production")
        assert manager.quality == 75

    def test_quality_clamping(self):
        """Quality should be clamped to 0-100."""
        manager = QualityManager(quality=-10)
        assert manager.quality == 0

        manager = QualityManager(quality=150)
        assert manager.quality == 100

    def test_quality_setter(self):
        """Quality setter should work."""
        manager = QualityManager(quality=50)
        manager.quality = 80
        assert manager.quality == 80

        manager.quality = "draft"
        assert manager.quality == 20

    def test_get_model_simple_task(self):
        """Model selection for simple tasks."""
        manager = QualityManager(quality=20)
        assert manager.get_model("simple") == "gpt-4o-mini"

        manager = QualityManager(quality=50)
        assert manager.get_model("simple") == "gpt-4o-mini"

        manager = QualityManager(quality=80)
        assert manager.get_model("simple") == "gpt-4o"

    def test_get_model_medium_task(self):
        """Model selection for medium tasks."""
        manager = QualityManager(quality=20)
        assert manager.get_model("medium") == "gpt-4o-mini"

        manager = QualityManager(quality=50)
        assert manager.get_model("medium") == "gpt-4o"

        manager = QualityManager(quality=80)
        assert manager.get_model("medium") == "gpt-4-turbo"

    def test_get_model_complex_task(self):
        """Model selection for complex tasks."""
        manager = QualityManager(quality=20)
        assert manager.get_model("complex") == "gpt-4o"

        manager = QualityManager(quality=50)
        assert manager.get_model("complex") == "gpt-4-turbo"

        manager = QualityManager(quality=80)
        assert manager.get_model("complex") == "gpt-4-turbo"

    def test_get_model_with_enum(self):
        """Model selection with TaskComplexity enum."""
        manager = QualityManager(quality=50)
        assert manager.get_model(TaskComplexity.SIMPLE) == "gpt-4o-mini"
        assert manager.get_model(TaskComplexity.MEDIUM) == "gpt-4o"
        assert manager.get_model(TaskComplexity.COMPLEX) == "gpt-4-turbo"

    def test_get_generation_params_low_quality(self):
        """Generation params for low quality."""
        manager = QualityManager(quality=20)
        params = manager.get_generation_params()

        assert params["max_tokens"] == 500
        assert params["temperature"] == 0.3

    def test_get_generation_params_medium_quality(self):
        """Generation params for medium quality."""
        manager = QualityManager(quality=50)
        params = manager.get_generation_params()

        assert params["max_tokens"] == 1000
        assert params["temperature"] == 0.5

    def test_get_generation_params_high_quality(self):
        """Generation params for high quality."""
        manager = QualityManager(quality=80)
        params = manager.get_generation_params()

        assert params["max_tokens"] == 2000
        assert params["temperature"] == 0.7

    def test_get_pipeline_depth(self):
        """Pipeline depth by quality."""
        manager = QualityManager(quality=20)
        assert manager.get_pipeline_depth() == PipelineDepth.SHALLOW

        manager = QualityManager(quality=50)
        assert manager.get_pipeline_depth() == PipelineDepth.MEDIUM

        manager = QualityManager(quality=80)
        assert manager.get_pipeline_depth() == PipelineDepth.DEEP

    def test_get_agent_count_range(self):
        """Agent count range by quality."""
        manager = QualityManager(quality=20)
        assert manager.get_agent_count_range() == (1, 2)

        manager = QualityManager(quality=50)
        assert manager.get_agent_count_range() == (2, 4)

        manager = QualityManager(quality=80)
        assert manager.get_agent_count_range() == (3, 6)

    def test_get_cost_multiplier(self):
        """Cost multiplier by quality."""
        manager = QualityManager(quality=0)
        assert manager.get_cost_multiplier() == 0.5

        manager = QualityManager(quality=50)
        assert manager.get_cost_multiplier() == pytest.approx(2.75)

        manager = QualityManager(quality=100)
        assert manager.get_cost_multiplier() == 5.0

    def test_estimate_cost(self):
        """Cost estimation."""
        manager = QualityManager(quality=50)
        min_cost, max_cost = manager.estimate_cost("medium")

        assert min_cost > 0
        assert max_cost > min_cost
        assert max_cost / min_cost == pytest.approx(1.3 / 0.7)

    def test_with_importance(self):
        """Quality adjustment by importance."""
        manager = QualityManager(quality=50)

        assert manager.with_importance("high") == 70
        assert manager.with_importance("medium") == 50
        assert manager.with_importance("low") == 30

    def test_with_importance_clamping(self):
        """Importance adjustment should clamp to 0-100."""
        manager = QualityManager(quality=90)
        assert manager.with_importance("high") == 100

        manager = QualityManager(quality=10)
        assert manager.with_importance("low") == 0

    def test_to_dict(self):
        """Export to dictionary."""
        manager = QualityManager(quality=70)
        data = manager.to_dict()

        assert data["quality"] == 70
        assert data["auto_mode"] is False
        assert "model_simple" in data
        assert "model_medium" in data
        assert "model_complex" in data
        assert "pipeline_depth" in data
        assert "cost_multiplier" in data

    def test_repr(self):
        """String representation."""
        manager = QualityManager(quality=70)
        assert "70" in repr(manager)
        assert "fixed" in repr(manager)


class TestQualityManagerAutoMode:
    """Tests for auto quality mode."""

    def test_auto_mode_init(self):
        """Auto mode initialization."""
        manager = QualityManager(quality="auto")
        assert manager._auto_mode is True
        assert manager.quality == 50  # Default when no budget set

    def test_auto_mode_with_budget(self):
        """Auto mode with budget tracking."""
        manager = QualityManager(quality="auto")
        manager.set_daily_budget(10.0)

        # Full budget -> high quality
        assert manager.quality == 70

        # Spend some
        manager.record_spend(5.0)
        assert manager.quality == 50

        # Spend more
        manager.record_spend(3.0)
        assert manager.quality == 30

        # Almost depleted
        manager.record_spend(1.5)
        assert manager.quality == 20

    def test_reset_daily_spend(self):
        """Reset daily spend counter."""
        manager = QualityManager(quality="auto")
        manager.set_daily_budget(10.0)
        manager.record_spend(8.0)

        assert manager.quality < 50

        manager.reset_daily_spend()
        assert manager.quality == 70


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_create(self):
        """Create CostEstimate."""
        estimate = CostEstimate(
            min_cost=0.10,
            max_cost=0.20,
            quality=70,
            task_complexity="medium",
        )

        assert estimate.min_cost == 0.10
        assert estimate.max_cost == 0.20
        assert estimate.quality == 70
        assert estimate.task_complexity == "medium"

    def test_average(self):
        """Average cost calculation."""
        estimate = CostEstimate(
            min_cost=0.10,
            max_cost=0.20,
            quality=70,
            task_complexity="medium",
        )

        assert estimate.average == pytest.approx(0.15)

    def test_to_dict(self):
        """Export to dictionary."""
        estimate = CostEstimate(
            min_cost=0.10,
            max_cost=0.20,
            quality=70,
            task_complexity="medium",
        )

        data = estimate.to_dict()
        assert data["min_cost"] == 0.10
        assert data["max_cost"] == 0.20
        assert data["average_cost"] == pytest.approx(0.15)
        assert data["quality"] == 70
        assert data["task_complexity"] == "medium"

    def test_str(self):
        """String representation."""
        estimate = CostEstimate(
            min_cost=0.10,
            max_cost=0.20,
            quality=70,
            task_complexity="medium",
        )

        assert str(estimate) == "$0.10 - $0.20"


class TestCostEstimator:
    """Tests for CostEstimator class."""

    def test_estimate_simple(self):
        """Estimate for simple task."""
        estimator = CostEstimator()
        estimate = estimator.estimate(quality=50, complexity="simple")

        assert estimate.quality == 50
        assert estimate.task_complexity == "simple"
        assert estimate.min_cost > 0
        assert estimate.max_cost > estimate.min_cost

    def test_estimate_medium(self):
        """Estimate for medium task."""
        estimator = CostEstimator()
        estimate = estimator.estimate(quality=50, complexity="medium")

        assert estimate.task_complexity == "medium"
        # Medium should cost more than simple
        simple = estimator.estimate(quality=50, complexity="simple")
        assert estimate.min_cost > simple.min_cost

    def test_estimate_complex(self):
        """Estimate for complex task."""
        estimator = CostEstimator()
        estimate = estimator.estimate(quality=50, complexity="complex")

        assert estimate.task_complexity == "complex"
        # Complex should cost more than medium
        medium = estimator.estimate(quality=50, complexity="medium")
        assert estimate.min_cost > medium.min_cost

    def test_estimate_quality_affects_cost(self):
        """Higher quality should cost more."""
        estimator = CostEstimator()

        low = estimator.estimate(quality=20, complexity="medium")
        high = estimator.estimate(quality=80, complexity="medium")

        assert high.min_cost > low.min_cost
        assert high.max_cost > low.max_cost

    def test_estimate_detailed(self):
        """Detailed estimation with agent configs."""
        estimator = CostEstimator()
        agents = [
            {"role": "extractor", "model": "gpt-4o-mini"},
            {"role": "analyzer", "model": "gpt-4o"},
            {"role": "writer", "model": "gpt-4o"},
        ]

        estimate = estimator.estimate_detailed(quality=50, agents=agents)

        assert estimate.task_complexity == "custom"
        assert estimate.breakdown is not None
        assert "extractor" in estimate.breakdown
        assert "analyzer" in estimate.breakdown
        assert "writer" in estimate.breakdown

    def test_suggest_quality_for_budget(self):
        """Suggest quality for budget."""
        estimator = CostEstimator()

        # Low budget -> low quality
        low_quality = estimator.suggest_quality_for_budget(0.05, "medium")
        assert low_quality < 50

        # High budget -> high quality
        high_quality = estimator.suggest_quality_for_budget(0.50, "medium")
        assert high_quality > 50

    def test_suggest_quality_clamping(self):
        """Quality suggestion should be clamped."""
        estimator = CostEstimator()

        # Very low budget
        quality = estimator.suggest_quality_for_budget(0.001, "medium")
        assert quality >= 0

        # Very high budget
        quality = estimator.suggest_quality_for_budget(100.0, "simple")
        assert quality <= 100
