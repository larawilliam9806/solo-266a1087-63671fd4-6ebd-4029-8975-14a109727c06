"""Retrieval-list planning.

For every archived event the reviewer needs each component (Z/N/E) to cover
the half-open window ``[origin - 30s, origin + 90s)``.  Given a station's
waveform index, the planner turns each window into either:

* the minimum set of files that continuously cover it, with one non-overlapping
  cut range ``[cut_start, cut_end)`` per file; or
* the uncovered sub-intervals ("missing segments") when the index cannot cover
  the whole window.

The minimum-file selection is the classic minimum interval-cover greedy
algorithm: repeatedly pick the available segment that starts no later than the
cursor and reaches farthest.  Ties are broken deterministically by file number
then start time so the same index always yields the same list.

Everything here is a pure function over integer epoch seconds; persistence and
HTTP layers live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .waveform import COMPONENTS, POST_SECONDS, PRE_SECONDS


@dataclass(frozen=True)
class Segment:
    file_no: str
    start_ts: int
    end_ts: int


@dataclass(frozen=True)
class Cut:
    """One half-open slice ``[start_ts, end_ts)`` taken from one file."""

    file_no: str
    start_ts: int
    end_ts: int


@dataclass
class ComponentPlan:
    component: str
    cuts: list[Cut] = field(default_factory=list)
    gaps: list[tuple[int, int]] = field(default_factory=list)

    @property
    def covered(self) -> bool:
        return not self.gaps


def plan_component(
    component: str,
    window_start: int,
    window_end: int,
    segments: list[Segment],
) -> ComponentPlan:
    """Plan one component against one event window.

    ``segments`` are index intervals of a single component; overlaps between
    index files are allowed.  The produced cut ranges never overlap.
    """
    target_start, target_end = window_start, window_end
    usable = [s for s in segments if s.start_ts < target_end and s.end_ts > target_start]
    # Farthest reach first; ties prefer the smaller file number then earlier
    # start so the output is deterministic.
    usable.sort(key=lambda s: (-s.end_ts, s.file_no, s.start_ts))

    cuts: list[Cut] = []
    gaps: list[tuple[int, int]] = []
    cursor = target_start
    gap_open: int | None = None

    while cursor < target_end:
        candidates = [s for s in usable if s.start_ts <= cursor and s.end_ts > cursor]
        if not candidates:
            if gap_open is None:
                gap_open = cursor
            # No file covers the cursor: advance to the next segment start.
            next_starts = [
                s.start_ts for s in usable if cursor < s.start_ts < target_end
            ]
            if not next_starts:
                gaps.append((gap_open, target_end))
                break
            cursor = min(next_starts)
            continue
        if gap_open is not None:
            gaps.append((gap_open, cursor))
            gap_open = None
        picked = candidates[0]
        cut_end = min(picked.end_ts, target_end)
        cuts.append(Cut(picked.file_no, cursor, cut_end))
        cursor = cut_end

    return ComponentPlan(
        component=component,
        cuts=cuts,
        gaps=gaps,
    )


def event_window(origin_ts: int) -> tuple[int, int]:
    """Half-open retrieval window for an event: [origin-30, origin+90)."""
    return origin_ts - PRE_SECONDS, origin_ts + POST_SECONDS


def plan_event(
    origin_ts: int,
    index_by_component: dict[str, list[Segment]],
    components: tuple[str, ...] = COMPONENTS,
) -> dict[str, ComponentPlan]:
    """Plan all components for one event."""
    start, end = event_window(origin_ts)
    plans: dict[str, ComponentPlan] = {}
    for component in components:
        plans[component] = plan_component(
            component, start, end, list(index_by_component.get(component, []))
        )
    return plans


def all_covered(plans: dict[str, ComponentPlan]) -> bool:
    return all(plan.covered for plan in plans.values())
