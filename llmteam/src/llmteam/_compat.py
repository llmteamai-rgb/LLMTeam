"""
Backward compatibility module for migration from llm_pipeline_smtrk to llmteam.

This module allows existing code using `llm_pipeline_smtrk` to continue working
with a deprecation warning for 2 releases.

Usage (old code - will show deprecation warning):
    from llm_pipeline_smtrk import create_pipeline
    
Usage (new code - recommended):
    from llmteam import create_pipeline
"""

import warnings
from functools import wraps
from typing import Any


_DEPRECATION_WARNING = (
    "The package 'llm_pipeline_smtrk' has been renamed to 'llmteam'. "
    "Please update your imports. This compatibility layer will be removed in v1.9.0."
)


def deprecated_import(name: str, obj: Any) -> Any:
    """
    Wrap an object to show deprecation warning on first use.
    """
    warnings.warn(
        f"Importing '{name}' from 'llm_pipeline_smtrk' is deprecated. {_DEPRECATION_WARNING}",
        DeprecationWarning,
        stacklevel=3,
    )
    return obj


class CompatibilityModule:
    """
    A module-like object that shows deprecation warnings on attribute access.
    """
    
    def __init__(self, new_module: Any):
        self._new_module = new_module
        self._warned = set()
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        
        if name not in self._warned:
            warnings.warn(
                f"Accessing '{name}' from deprecated module. {_DEPRECATION_WARNING}",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned.add(name)
        
        return getattr(self._new_module, name)
    
    def __dir__(self):
        return dir(self._new_module)


def create_compatibility_layer():
    """
    Create the compatibility layer for llm_pipeline_smtrk.
    
    This function should be called during package initialization to set up
    the backward compatibility module.
    """
    import sys
    import llmteam
    
    # Create compatibility module
    compat = CompatibilityModule(llmteam)
    
    # Register under old name
    sys.modules['llm_pipeline_smtrk'] = compat
    
    return compat


# Migration helpers

def check_deprecated_usage() -> list[str]:
    """
    Check for deprecated usage patterns in the current codebase.
    
    Returns:
        List of deprecation warnings found.
    """
    import sys
    
    warnings_found = []
    
    # Check if old module name is in sys.modules
    if 'llm_pipeline_smtrk' in sys.modules:
        warnings_found.append(
            "Found import of deprecated module 'llm_pipeline_smtrk'"
        )
    
    return warnings_found


def migration_guide() -> str:
    """
    Return a migration guide for updating from llm_pipeline_smtrk to llmteam.
    """
    return """
Migration Guide: llm_pipeline_smtrk → llmteam
=============================================

1. Update your imports:
   
   # Before
   from llm_pipeline_smtrk import create_pipeline
   
   # After
   from llmteam import create_pipeline

2. Update your requirements.txt or pyproject.toml:
   
   # Before
   llm-pipeline-smtrk>=1.6.0
   
   # After
   llmteam>=1.7.0

3. Update any configuration files that reference the old package name.

4. Run your tests to ensure everything works correctly.

5. The compatibility layer will be removed in v1.9.0, so please update
   your code before then.

New Features in v1.7.0:
- Multi-tenant isolation (TenantManager, TenantContext)
- Audit trail for compliance (AuditTrail, AuditRecord)
- Context security (SecureAgentContext, SealedData)
- Rate limiting (RateLimiter, CircuitBreaker)

For more information, see the documentation at:
https://llmteam.readthedocs.io/migration
"""
