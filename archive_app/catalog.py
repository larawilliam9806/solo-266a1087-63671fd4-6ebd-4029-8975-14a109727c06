"""Local earthquake catalog archive service.

The application talks to a *catalog source* through a small, well-defined
interface.  The bundled ``MockCatalogSource`` simulates the regional network's
paged-by-date endpoint (max 20 events per response).  A real deployment can
swap in another source with the same two methods.
"""

from __future__ import annotations

import abc
import datetime as dt
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

from .planner import Segment, event_window, plan_event
from .waveform import COMPONENTS, IndexRow, parse_wave_time


PAGE_SIZE = 20
MAX_SPAN_DAYS = 31
UTC = dt.timezone.utc


class SourceError(Exception):
    """Raised when the upstream catalog cannot be queried."""


@dataclass(frozen=True)
class Event:
    event_id: str
    utc_time: str  # canonical "YYYY-MM-DD HH:MM:SS"
    magnitude: float

    def as_tuple(self) -> tuple[str, str, float]:
        return (self.event_id, self.utc_time, self.magnitude)


def canonical_time(value: str) -> str:
    """Normalise a UTC issuance time to ``YYYY-MM-DD HH:MM:SS``."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"无法解析 UTC 时刻: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def parse_day(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"日期必须为 YYYY-MM-DD: {value!r}") from exc


class CatalogSource(abc.ABC):
    @abc.abstractmethod
    def fetch_page(self, start: dt.date, end: dt.date) -> list[Event]:
        """Return the events with start <= UTC date <= end, capped at 20.

        The upstream has no paging: a full response means "there may be more
        inside this window", which the collector resolves by narrowing dates.
        """

    def page_size(self) -> int:
        return PAGE_SIZE


class MockCatalogSource(CatalogSource):
    """Reads a deterministic JSON catalog standing in for the live endpoint.

    File layout::

        {"events": [{"id": "...", "time": "2024-01-02T03:04:05Z",
                     "magnitude": 3.2}, ...]}

    The mock is intentionally strict: it only serves windows of at most 31
    days, returns at most 20 events, and reports an error for a configured list
    of dates so retry logic can be exercised.
    """

    def __init__(self, path: str | os.PathLike[str], fail_dates: Iterable[str] = ()):
        self.path = os.fspath(path)
        self.fail_dates = set(fail_dates)
        self._lock = threading.Lock()
        self._events: list[Event] | None = None
        self._mtime: float | None = None

    def _load(self) -> list[Event]:
        mtime = os.stat(self.path).st_mtime
        with self._lock:
            if self._events is None or mtime != self._mtime:
                with open(self.path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                events = []
                for raw in payload.get("events", []):
                    events.append(
                        Event(
                            event_id=str(raw["id"]).strip(),
                            utc_time=canonical_time(raw["time"]),
                            magnitude=float(raw["magnitude"]),
                        )
                    )
                events.sort(key=lambda e: (e.utc_time, e.event_id))
                self._events = events
                self._mtime = mtime
            return self._events

    def fetch_page(self, start: dt.date, end: dt.date) -> list[Event]:
        if end < start:
            raise SourceError("日期区间无效")
        if (end - start).days + 1 > MAX_SPAN_DAYS:
            raise SourceError(f"日期区间不得超过 {MAX_SPAN_DAYS} 天")
        for day in self.fail_dates:
            d = dt.date.fromisoformat(day)
            if start <= d <= end:
                raise SourceError(f"模拟接口故障: {day}")
        events = [
            e
            for e in self._load()
            if start <= dt.date.fromisoformat(e.utc_time[:10]) <= end
        ]
        return events[:PAGE_SIZE]


def source_from_env() -> CatalogSource:
    path = os.environ.get("CATALOG_FILE", "")
    if not path:
        for candidate in ("data/catalog.json", "seed/catalog.json"):
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            path = "data/catalog.json"
    if not os.path.exists(path):
        raise RuntimeError(
            f"找不到模拟目录数据文件 {path}；用 CATALOG_FILE 指定，"
            "或运行 scripts/generate_catalog.py 生成。"
        )
    fail = [d.strip() for d in os.environ.get("MOCK_FAIL_DATES", "").split(",") if d.strip()]
    return MockCatalogSource(path, fail_dates=fail)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS day_status (
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    checked_at TEXT,
    last_error TEXT,
    PRIMARY KEY (run_id, day),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS events (
    run_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    event_id TEXT NOT NULL,
    utc_time TEXT NOT NULL,
    magnitude REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(run_id, day);

-- A waveform index belongs to a station and a UTC day.  Each row is one
-- component interval of a physical file; (station, file_no) identifies a file.
CREATE TABLE IF NOT EXISTS waveform_index (
    station TEXT NOT NULL,
    day TEXT NOT NULL,
    component TEXT NOT NULL,
    file_no TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    PRIMARY KEY (station, day, file_no, component)
);
CREATE INDEX IF NOT EXISTS idx_wave_station_time
    ON waveform_index(station, start_ts, end_ts);
CREATE INDEX IF NOT EXISTS idx_wave_file ON waveform_index(station, file_no);

-- One retrieval list per (run, station).  Enabled lists are kept in sync
-- atomically whenever a day becomes complete or the station index changes.
CREATE TABLE IF NOT EXISTS retrieval_lists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    station TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, station),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

-- Saved planner output: one row per event/component, with the cut ranges and
-- missing gaps stored as JSON arrays of [start_ts, end_ts] pairs.
CREATE TABLE IF NOT EXISTS retrieval_windows (
    list_id INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    event_time TEXT NOT NULL,
    origin_ts INTEGER NOT NULL,
    window_start INTEGER NOT NULL,
    window_end INTEGER NOT NULL,
    component TEXT NOT NULL,
    covered INTEGER NOT NULL,
    cuts TEXT NOT NULL,
    gaps TEXT NOT NULL,
    PRIMARY KEY (list_id, event_id, component),
    FOREIGN KEY (list_id) REFERENCES retrieval_lists(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_windows_list ON retrieval_windows(list_id, origin_ts, event_id);
"""

LIST_DISABLED = 0
LIST_ENABLED = 1

STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"   # day is fully archived
STATUS_BLOCKED = "blocked"     # day returned a full page; needs CSV backfill
STATUS_FAILED = "failed"       # request failed; retry later

ALLOWED_TRANSITIONS = {
    STATUS_PENDING: {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED},
    STATUS_FAILED: {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED},
    STATUS_BLOCKED: {STATUS_COMPLETE, STATUS_BLOCKED},
    STATUS_COMPLETE: set(),
}


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA)

    # -- runs ---------------------------------------------------------

    def create_run(self, start: dt.date, end: dt.date) -> int:
        with self._lock, self.connect() as conn:
            run_id = self._create_run(conn, start, end)
            conn.commit()
            return run_id

    @staticmethod
    def _create_run(conn: sqlite3.Connection, start: dt.date, end: dt.date) -> int:
        now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = conn.execute(
            "INSERT INTO runs(start_date, end_date, created_at) VALUES (?,?,?)",
            (start.isoformat(), end.isoformat(), now),
        )
        run_id = int(cur.lastrowid)
        day = start
        while day <= end:
            conn.execute(
                "INSERT INTO day_status(run_id, day, status) VALUES (?,?,?)",
                (run_id, day.isoformat(), STATUS_PENDING),
            )
            day += dt.timedelta(days=1)
        return run_id

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def runs(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT r.*, "
                "(SELECT COUNT(*) FROM day_status d WHERE d.run_id=r.id AND d.status='complete') AS complete_days, "
                "(SELECT COUNT(*) FROM day_status d WHERE d.run_id=r.id AND d.status='blocked') AS blocked_days "
                "FROM runs r ORDER BY r.id DESC"
            ).fetchall()

    def days(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM day_status WHERE run_id=? ORDER BY day", (run_id,)
            ).fetchall()

    def events(self, run_id: int, day: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as conn:
            if day is None:
                return conn.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY utc_time, event_id",
                    (run_id,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM events WHERE run_id=? AND day=? ORDER BY utc_time, event_id",
                (run_id, day),
            ).fetchall()

    # -- progress -----------------------------------------------------

    def mark(self, run_id: int, day: str, status: str, last_error: str | None = None) -> None:
        if status not in {STATUS_COMPLETE, STATUS_BLOCKED, STATUS_FAILED}:
            raise ValueError(status)
        with self._lock, self.connect() as conn:
            self._mark(conn, run_id, day, status, last_error)
            conn.commit()

    @staticmethod
    def _mark(
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        status: str,
        last_error: str | None = None,
    ) -> None:
        row = conn.execute(
            "SELECT status FROM day_status WHERE run_id=? AND day=?", (run_id, day)
        ).fetchone()
        if row is None:
            raise KeyError(day)
        current = row["status"]
        if current not in ALLOWED_TRANSITIONS or status not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"状态不允许从 {current} 变为 {status}")
        now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=? AND day=?", (run_id, day)
        ).fetchone()[0]
        conn.execute(
            "UPDATE day_status SET status=?, event_count=?, checked_at=?, last_error=? "
            "WHERE run_id=? AND day=?",
            (status, count, now, last_error, run_id, day),
        )

    def complete_days(
        self,
        run_id: int,
        by_day: dict[str, Sequence[Event]],
        source: str = "api",
    ) -> None:
        """Persist a whole exhausted range and mark its days complete at once.

        The write is transactional: any conflict (for example the same event ID
        showing up on two days) rolls the range back so no day is half-recorded.
        """
        with self._lock, self.connect() as conn:
            try:
                for day, events in by_day.items():
                    for e in events:
                        if e.utc_time[:10] != day:
                            raise ValueError(f"事件 {e.event_id} 不在 {day}")
                        try:
                            conn.execute(
                                "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                                "VALUES (?,?,?,?,?,?)",
                                (run_id, day, e.event_id, e.utc_time, e.magnitude, source),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"事件编号冲突: {e.event_id}") from exc
                    self._mark(conn, run_id, day, STATUS_COMPLETE)
                # Enabled retrieval lists take effect together with the newly
                # complete days, inside the same transaction.
                self._sync_enabled_lists(conn, run_id, list(by_day))
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    def record_blocked_day(self, run_id: int, day: str, events: Sequence[Event]) -> None:
        """Store the first (full) page for a day that cannot be exhausted."""
        with self._lock, self.connect() as conn:
            try:
                for e in events:
                    try:
                        conn.execute(
                            "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                            "VALUES (?,?,?,?,?,?)",
                            (run_id, day, e.event_id, e.utc_time, e.magnitude, "api"),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"事件编号冲突: {e.event_id}") from exc
                self._mark(conn, run_id, day, STATUS_BLOCKED)
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    @staticmethod
    def _replace(
        conn: sqlite3.Connection,
        run_id: int,
        day: str,
        events: Sequence[Event],
        source: str,
    ) -> None:
        conn.execute("DELETE FROM events WHERE run_id=? AND day=?", (run_id, day))
        for e in events:
            try:
                conn.execute(
                    "INSERT INTO events(run_id, day, event_id, utc_time, magnitude, source) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, day, e.event_id, e.utc_time, e.magnitude, source),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"事件编号已被其他日期使用: {e.event_id}") from exc

    def backfill_day(
        self,
        run_id: int,
        day: str,
        events: Sequence[Event],
        verify=None,
    ) -> None:
        """Atomically replace a blocked day's events and mark it complete.

        ``verify`` receives the already-collected events and may raise to abort;
        the whole operation rolls back in that case, preserving prior progress.
        """
        with self._lock, self.connect() as conn:
            row = conn.execute(
                "SELECT status FROM day_status WHERE run_id=? AND day=?", (run_id, day)
            ).fetchone()
            if row is None:
                raise KeyError(day)
            if row["status"] != STATUS_BLOCKED:
                raise ValueError(f"{day} 不是待补档日期")
            existing = [
                Event(r["event_id"], r["utc_time"], r["magnitude"])
                for r in conn.execute(
                    "SELECT event_id, utc_time, magnitude FROM events "
                    "WHERE run_id=? AND day=? ORDER BY utc_time, event_id",
                    (run_id, day),
                )
            ]
            if verify is not None:
                verify(existing)
            self._replace(conn, run_id, day, events, "csv")
            self._mark(conn, run_id, day, STATUS_COMPLETE)
            # Backfilled day + enabled lists become effective together.
            self._sync_enabled_lists(conn, run_id, [day])
            conn.commit()

    def is_ready(self, run_id: int) -> bool:
        with self.connect() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM day_status WHERE run_id=? AND status!='complete'",
                (run_id,),
            ).fetchone()[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM day_status WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            return total > 0 and bad == 0

    # ------------------------------------------------------------------
    # Waveform index
    # ------------------------------------------------------------------

    def stations(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT station FROM waveform_index ORDER BY station"
            ).fetchall()
        return [r["station"] for r in rows]

    def index_days(self, station: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT day, COUNT(*) AS segment_count, "
                "COUNT(DISTINCT file_no) AS file_count, "
                "MIN(start_ts) AS min_ts, MAX(end_ts) AS max_ts "
                "FROM waveform_index WHERE station=? GROUP BY day ORDER BY day",
                (station,),
            ).fetchall()

    def index_rows(self, station: str, day: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM waveform_index WHERE station=? AND day=? "
                "ORDER BY file_no, component",
                (station, day),
            ).fetchall()

    def station_index_segments(
        self, conn: sqlite3.Connection, station: str
    ) -> dict[str, list[Segment]]:
        rows = conn.execute(
            "SELECT component, file_no, start_ts, end_ts FROM waveform_index "
            "WHERE station=? ORDER BY start_ts, file_no",
            (station,),
        ).fetchall()
        grouped: dict[str, list[Segment]] = {c: [] for c in COMPONENTS}
        for row in rows:
            grouped.setdefault(row["component"], []).append(
                Segment(row["file_no"], row["start_ts"], row["end_ts"])
            )
        return grouped

    def replace_index(
        self, station: str, day: str, rows: Sequence[IndexRow]
    ) -> dict[str, int]:
        """Replace one station-day index, recomputing only intersecting windows.

        The whole operation (delete, insert, incremental list recomputation) is
        transactional.  Validation against other days' files happens before any
        write, so a failure leaves the prior index and every retrieval list
        untouched.  Recomputation is limited to this station and to event
        windows whose time range intersects a swapped-in or swapped-out file.
        """
        with self._lock, self.connect() as conn:
            try:
                old = conn.execute(
                    "SELECT file_no, start_ts, end_ts FROM waveform_index "
                    "WHERE station=? AND day=?",
                    (station, day),
                ).fetchall()
                # File numbers are unique within a station: a number used on
                # another day cannot be reused.
                other_days = {
                    r["file_no"]
                    for r in conn.execute(
                        "SELECT DISTINCT file_no FROM waveform_index "
                        "WHERE station=? AND day!=?",
                        (station, day),
                    ).fetchall()
                }
                new_files = {r.file_no for r in rows}
                conflicts = sorted(other_days & new_files)
                if conflicts:
                    raise ValueError(
                        "文件编号已在该台站其他日期使用: " + ", ".join(conflicts)
                    )

                # Windows touching any old or new file interval need replanning.
                bounds = [
                    (r["start_ts"], r["end_ts"])
                    for r in old
                ] + [(r.start_ts, r.end_ts) for r in rows]
                lo = min((b[0] for b in bounds), default=None)
                hi = max((b[1] for b in bounds), default=None)

                conn.execute(
                    "DELETE FROM waveform_index WHERE station=? AND day=?",
                    (station, day),
                )
                for row in rows:
                    conn.execute(
                        "INSERT INTO waveform_index"
                        "(station, day, component, file_no, start_ts, end_ts) "
                        "VALUES (?,?,?,?,?,?)",
                        (
                            station,
                            day,
                            row.component,
                            row.file_no,
                            row.start_ts,
                            row.end_ts,
                        ),
                    )

                recomputed = 0
                if lo is not None:
                    recomputed = self._recompute_windows_span(conn, station, lo, hi)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
        return {"rows": len(rows), "windows_recomputed": recomputed}

    # ------------------------------------------------------------------
    # Retrieval lists
    # ------------------------------------------------------------------

    def lists_for_run(self, run_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT l.*, "
                "(SELECT COUNT(*) FROM retrieval_windows w WHERE w.list_id=l.id) AS window_rows, "
                "(SELECT COUNT(DISTINCT w.event_id) FROM retrieval_windows w WHERE w.list_id=l.id) AS event_count, "
                "(SELECT MIN(w.covered) FROM retrieval_windows w WHERE w.list_id=l.id) AS all_covered "
                "FROM retrieval_lists l WHERE l.run_id=? ORDER BY l.station",
                (run_id,),
            ).fetchall()

    def get_list(self, list_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT l.*, r.start_date AS run_start, r.end_date AS run_end "
                "FROM retrieval_lists l JOIN runs r ON r.id=l.run_id "
                "WHERE l.id=?",
                (list_id,),
            ).fetchone()

    def list_windows(self, list_id: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM retrieval_windows WHERE list_id=? "
                "ORDER BY origin_ts, event_id, component",
                (list_id,),
            ).fetchall()

    def generate_retrieval_list(self, run_id: int, station: str) -> int:
        """(Re)build the run/station list from completed-day events.

        Replaces every saved window of the list in one transaction; progress,
        index and prior list content survive if anything fails.
        """
        with self._lock, self.connect() as conn:
            try:
                if conn.execute(
                    "SELECT 1 FROM runs WHERE id=?", (run_id,)
                ).fetchone() is None:
                    raise KeyError(run_id)
                now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                conn.execute(
                    "INSERT INTO retrieval_lists(run_id, station, enabled, updated_at) "
                    "VALUES (?,?,0,?) "
                    "ON CONFLICT(run_id, station) DO UPDATE SET updated_at=excluded.updated_at",
                    (run_id, station, now),
                )
                list_id = conn.execute(
                    "SELECT id FROM retrieval_lists WHERE run_id=? AND station=?",
                    (run_id, station),
                ).fetchone()["id"]
                conn.execute(
                    "DELETE FROM retrieval_windows WHERE list_id=?", (list_id,)
                )
                self._rebuild_windows(conn, list_id)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
        return list_id

    def set_list_enabled(self, list_id: int, enabled: bool) -> None:
        """Enable/disable a list; enabling refreshes it inside one transaction."""
        with self._lock, self.connect() as conn:
            try:
                row = conn.execute(
                    "SELECT run_id FROM retrieval_lists WHERE id=?", (list_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(list_id)
                now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                conn.execute(
                    "UPDATE retrieval_lists SET enabled=?, updated_at=? WHERE id=?",
                    (LIST_ENABLED if enabled else LIST_DISABLED, now, list_id),
                )
                if enabled:
                    conn.execute(
                        "DELETE FROM retrieval_windows WHERE list_id=?", (list_id,)
                    )
                    self._rebuild_windows(conn, list_id)
            except Exception:
                conn.rollback()
                raise
            conn.commit()

    # -- planner plumbing ----------------------------------------------

    @staticmethod
    def _completed_events(
        conn: sqlite3.Connection, run_id: int, days: Sequence[str] | None = None
    ) -> list[sqlite3.Row]:
        sql = (
            "SELECT e.event_id, e.utc_time FROM events e "
            "JOIN day_status d ON d.run_id=e.run_id AND d.day=e.day "
            "WHERE e.run_id=? AND d.status='complete'"
        )
        params: list = [run_id]
        if days is not None:
            if not days:
                return []
            sql += " AND e.day IN (%s)" % ",".join("?" for _ in days)
            params.extend(days)
        sql += " ORDER BY e.utc_time, e.event_id"
        return conn.execute(sql, params).fetchall()

    def _rebuild_windows(self, conn: sqlite3.Connection, list_id: int) -> int:
        meta = conn.execute(
            "SELECT run_id, station FROM retrieval_lists WHERE id=?", (list_id,)
        ).fetchone()
        if meta is None:
            raise KeyError(list_id)
        segments = self.station_index_segments(conn, meta["station"])
        events = self._completed_events(conn, meta["run_id"])
        return self._write_windows(conn, list_id, segments, events)

    def _sync_enabled_lists(
        self, conn: sqlite3.Connection, run_id: int, days: Sequence[str]
    ) -> None:
        """Replan newly completed-day events for every enabled list of the run.

        Called inside the same transaction that marks the days complete, so an
        event and its retrieval windows appear together.
        """
        lists = conn.execute(
            "SELECT id, station FROM retrieval_lists WHERE run_id=? AND enabled=?",
            (run_id, LIST_ENABLED),
        ).fetchall()
        if not lists:
            return
        events = self._completed_events(conn, run_id, list(days))
        if not events:
            return
        for item in lists:
            segments = self.station_index_segments(conn, item["station"])
            event_ids = [e["event_id"] for e in events]
            conn.execute(
                "DELETE FROM retrieval_windows WHERE list_id=? AND event_id IN (%s)"
                % ",".join("?" for _ in event_ids),
                [item["id"], *event_ids],
            )
            self._write_windows(conn, item["id"], segments, events)

    def _recompute_windows_span(
        self, conn: sqlite3.Connection, station: str, lo: int, hi: int
    ) -> int:
        """Replan windows intersecting [lo, hi) for all lists of a station.

        Used after an index replacement: only windows that touch a swapped-in or
        swapped-out file interval are touched, and only lists of this station.
        """
        lists = conn.execute(
            "SELECT id FROM retrieval_lists WHERE station=?", (station,)
        ).fetchall()
        if not lists:
            return 0
        segments = self.station_index_segments(conn, station)
        count = 0
        for item in lists:
            affected = conn.execute(
                "SELECT DISTINCT event_id, event_time AS utc_time, origin_ts "
                "FROM retrieval_windows WHERE list_id=? "
                "AND window_start < ? AND window_end > ?",
                (item["id"], hi, lo),
            ).fetchall()
            if not affected:
                continue
            ids = [r["event_id"] for r in affected]
            conn.execute(
                "DELETE FROM retrieval_windows WHERE list_id=? AND event_id IN (%s)"
                % ",".join("?" for _ in ids),
                [item["id"], *ids],
            )
            count += self._write_windows(conn, item["id"], segments, list(affected))
        return count

    def _write_windows(
        self,
        conn: sqlite3.Connection,
        list_id: int,
        segments: dict[str, list[Segment]],
        events: Iterable[sqlite3.Row],
    ) -> int:
        written = 0
        for event in events:
            origin_ts = parse_wave_time(event["utc_time"])
            win_start, win_end = event_window(origin_ts)
            plans = plan_event(origin_ts, segments)
            for component in COMPONENTS:
                plan = plans[component]
                cuts = [
                    {"file_no": c.file_no, "start_ts": c.start_ts, "end_ts": c.end_ts}
                    for c in plan.cuts
                ]
                gaps = [[a, b] for a, b in plan.gaps]
                conn.execute(
                    "INSERT INTO retrieval_windows"
                    "(list_id, event_id, event_time, origin_ts, window_start, "
                    "window_end, component, covered, cuts, gaps) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        list_id,
                        event["event_id"],
                        event["utc_time"],
                        origin_ts,
                        win_start,
                        win_end,
                        component,
                        1 if plan.covered else 0,
                        json.dumps(cuts, separators=(",", ":")),
                        json.dumps(gaps, separators=(",", ":")),
                    ),
                )
                written += 1
        return written
