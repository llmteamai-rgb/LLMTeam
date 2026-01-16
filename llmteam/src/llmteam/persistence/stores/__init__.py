"""
Persistence Stores.

Provides storage implementations for snapshots:
- SnapshotStore base class
- MemorySnapshotStore for testing
"""

from .base import SnapshotStore
from .memory import MemorySnapshotStore

__all__ = ["SnapshotStore", "MemorySnapshotStore"]
