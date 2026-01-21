"""
Built-in Step Handlers.

This module provides handlers for all built-in step types.
"""

from llmteam.engine.handlers.llm_handler import LLMAgentHandler
from llmteam.engine.handlers.http_handler import HTTPActionHandler
from llmteam.engine.handlers.transform_handler import TransformHandler
from llmteam.engine.handlers.condition_handler import ConditionHandler
from llmteam.engine.handlers.parallel_handler import ParallelSplitHandler, ParallelJoinHandler

# Re-export HumanTaskHandler from original location for backwards compatibility
from llmteam.engine.handlers.human_handler import HumanTaskHandler, create_human_task_handler

# v2.0.4: New handlers
from llmteam.engine.handlers.loop_handler import LoopHandler
from llmteam.engine.handlers.error_handler import ErrorHandler, TryCatchHandler

# v2.2.0: New handlers
from llmteam.engine.handlers.subworkflow_handler import SubworkflowHandler
from llmteam.engine.handlers.switch_handler import SwitchHandler

__all__ = [
    "LLMAgentHandler",
    "HTTPActionHandler",
    "TransformHandler",
    "ConditionHandler",
    "ParallelSplitHandler",
    "ParallelJoinHandler",
    "HumanTaskHandler",
    "create_human_task_handler",
    # v2.0.4
    "LoopHandler",
    "ErrorHandler",
    "TryCatchHandler",
    # v2.2.0
    "SubworkflowHandler",
    "SwitchHandler",
]
