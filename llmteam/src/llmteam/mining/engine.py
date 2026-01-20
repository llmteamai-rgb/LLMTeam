"""
Process mining engine for llmteam.

Provides process discovery, metrics calculation, and XES export for workflow analysis.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List


def generate_uuid() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


@dataclass
class ProcessEvent:
    """
    Event for process mining.

    Attributes:
        event_id: Unique event identifier
        timestamp: When the event occurred
        activity: Activity name
        resource: Resource that performed the activity (e.g., agent name)
        case_id: Case identifier (e.g., run_id)
        lifecycle: Lifecycle state (start, complete, suspend, resume)
        duration_ms: Duration in milliseconds
        cost: Cost of the activity
        attributes: Additional custom attributes
    """

    event_id: str
    timestamp: datetime
    activity: str
    resource: str
    case_id: str
    lifecycle: str = "complete"
    duration_ms: int = 0
    cost: float = 0.0
    attributes: Dict = field(default_factory=dict)


@dataclass
class ProcessMetrics:
    """
    Metrics for a process.

    Attributes:
        avg_duration: Average case duration
        min_duration: Minimum case duration
        max_duration: Maximum case duration
        cases_per_hour: Throughput (cases per hour)
        completion_rate: Completion rate (0.0-1.0)
        error_rate: Error rate (0.0-1.0)
        retry_rate: Retry rate (0.0-1.0)
        bottleneck_activities: List of bottleneck activities
        waiting_time_by_activity: Waiting time for each activity
    """

    avg_duration: timedelta
    min_duration: timedelta
    max_duration: timedelta
    cases_per_hour: float
    completion_rate: float
    error_rate: float
    retry_rate: float
    bottleneck_activities: List[str]
    waiting_time_by_activity: Dict[str, timedelta]


@dataclass
class ProcessModel:
    """
    Discovered process model.

    Attributes:
        activities: List of activities in the process
        transitions: Possible transitions between activities
        frequencies: Frequency of each transition
        fitness: Model fitness (0.0-1.0)
        precision: Model precision (0.0-1.0)
    """

    activities: List[str]
    transitions: Dict[str, List[str]]
    frequencies: Dict[str, int]
    fitness: float = 1.0
    precision: float = 1.0


from llmteam.licensing import professional_only


@professional_only
class ProcessMiningEngine:
    """
    Engine for process mining and workflow analysis.

    Features:
    - Event recording
    - Process model discovery (simplified Alpha Miner)
    - Metrics calculation
    - XES export for ProM/Celonis

    Example:
        engine = ProcessMiningEngine()

        # Record events
        engine.record_event(ProcessEvent(
            event_id="evt_1",
            timestamp=datetime.now(),
            activity="validate_input",
            resource="agent_1",
            case_id="run_123",
        ))

        # Discover process model
        model = engine.discover_model()

        # Calculate metrics
        metrics = engine.calculate_metrics()

        # Export to XES
        xes = engine.export_xes()
    """

    def __init__(self):
        """Initialize process mining engine."""
        self._events: List[ProcessEvent] = []
        self._cases: Dict[str, List[ProcessEvent]] = {}

    def record_event(self, event: ProcessEvent) -> None:
        """
        Record a process event.

        Args:
            event: ProcessEvent to record
        """
        self._events.append(event)

        if event.case_id not in self._cases:
            self._cases[event.case_id] = []
        self._cases[event.case_id].append(event)

    def discover_model(self) -> ProcessModel:
        """
        Discover process model using simplified Alpha Miner.

        Returns:
            ProcessModel with activities and transitions
        """
        activities = set()
        transitions = {}
        frequencies = {}

        for case_events in self._cases.values():
            sorted_events = sorted(case_events, key=lambda e: e.timestamp)

            for i, event in enumerate(sorted_events):
                activities.add(event.activity)

                if i > 0:
                    prev = sorted_events[i - 1].activity
                    curr = event.activity

                    if prev not in transitions:
                        transitions[prev] = []
                    if curr not in transitions[prev]:
                        transitions[prev].append(curr)

                    key = f"{prev}->{curr}"
                    frequencies[key] = frequencies.get(key, 0) + 1

        return ProcessModel(
            activities=list(activities),
            transitions=transitions,
            frequencies=frequencies,
        )

    def calculate_metrics(self) -> ProcessMetrics:
        """
        Calculate process metrics.

        Returns:
            ProcessMetrics with performance indicators
        """
        durations = []
        errors = 0
        retries = 0

        for case_events in self._cases.values():
            if len(case_events) >= 2:
                start = min(e.timestamp for e in case_events)
                end = max(e.timestamp for e in case_events)
                durations.append(end - start)

            for event in case_events:
                if "error" in event.activity.lower():
                    errors += 1
                if "retry" in event.activity.lower():
                    retries += 1

        total_cases = len(self._cases)

        # Calculate duration statistics
        if durations:
            avg_duration = sum(durations, timedelta()) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = timedelta()

        # Calculate throughput
        if self._events:
            time_span = (datetime.now() - self._events[0].timestamp).total_seconds()
            cases_per_hour = total_cases / max(1, time_span / 3600)
        else:
            cases_per_hour = 0

        # Calculate completion rate
        completed_cases = sum(
            1 for c in self._cases.values()
            if any(e.lifecycle == "complete" for e in c)
        )
        completion_rate = completed_cases / max(1, total_cases)

        # Calculate error and retry rates
        total_events = len(self._events)
        error_rate = errors / max(1, total_events)
        retry_rate = retries / max(1, total_events)

        return ProcessMetrics(
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            cases_per_hour=cases_per_hour,
            completion_rate=completion_rate,
            error_rate=error_rate,
            retry_rate=retry_rate,
            bottleneck_activities=self._find_bottlenecks(),
            waiting_time_by_activity={},
        )

    def _find_bottlenecks(self) -> List[str]:
        """
        Find bottleneck activities based on duration.

        Returns:
            List of top 3 bottleneck activities
        """
        activity_durations = {}

        for event in self._events:
            if event.activity not in activity_durations:
                activity_durations[event.activity] = []
            activity_durations[event.activity].append(event.duration_ms)

        # Sort by average duration
        avg_durations = {
            activity: sum(durations) / len(durations)
            for activity, durations in activity_durations.items()
        }

        sorted_activities = sorted(
            avg_durations.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [activity for activity, _ in sorted_activities[:3]]

    def export_xes(self) -> str:
        """
        Export events to XES format for ProM/Celonis.

        Returns:
            XES XML string
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<log>')

        for case_id, events in self._cases.items():
            lines.append('  <trace>')
            lines.append(f'    <string key="concept:name" value="{case_id}"/>')

            for event in sorted(events, key=lambda e: e.timestamp):
                lines.append('    <event>')
                lines.append(f'      <string key="concept:name" value="{event.activity}"/>')
                lines.append(f'      <string key="org:resource" value="{event.resource}"/>')
                lines.append(f'      <date key="time:timestamp" value="{event.timestamp.isoformat()}"/>')
                lines.append(f'      <string key="lifecycle:transition" value="{event.lifecycle}"/>')

                # Add custom attributes
                for attr_key, attr_value in event.attributes.items():
                    lines.append(f'      <string key="{attr_key}" value="{attr_value}"/>')

                lines.append('    </event>')

            lines.append('  </trace>')

        lines.append('</log>')
        return '\n'.join(lines)

    def get_event_count(self) -> int:
        """
        Get total event count.

        Returns:
            Number of recorded events
        """
        return len(self._events)

    def get_case_count(self) -> int:
        """
        Get total case count.

        Returns:
            Number of unique cases
        """
        return len(self._cases)

    def clear(self) -> None:
        """Clear all events and cases."""
        self._events.clear()
        self._cases.clear()
