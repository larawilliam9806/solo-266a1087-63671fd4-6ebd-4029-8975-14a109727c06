"""Waveform index CSV parsing and validation.

A waveform index lists, for one station and one UTC day, the waveform files
held in cold storage.  Every data row records a component (Z/N/E), a file
number (unique within the station) and the half-open UTC interval
``[start_ts, end_ts)`` of whole seconds that the file covers for that
component.  Files start on the selected day, may span at most 24 hours and
may cross midnight.

A CSV that contains only a header is valid and means "clear this day's
index".
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from dataclasses import dataclass

UTC = dt.timezone.utc
DAY_SECONDS = 24 * 3600
MAX_FILE_SECONDS = DAY_SECONDS
COMPONENTS = ("Z", "N", "E")

# Retrieval windows open 30 s before origin time and close 90 s after.
PRE_SECONDS = 30
POST_SECONDS = 90

COMP_NAMES = {"component", "comp", "channel", "分量", "分量代码", "通道"}
FILE_NAMES = {
    "file_no",
    "fileno",
    "file",
    "file_id",
    "fileid",
    "文件编号",
    "文件号",
    "编号",
}
START_NAMES = {
    "start",
    "start_time",
    "start_utc",
    "start_ts",
    "开始",
    "起始",
    "开始时刻",
    "起始时刻",
}
END_NAMES = {
    "end",
    "end_time",
    "end_utc",
    "end_ts",
    "结束",
    "结束时刻",
    "截止时刻",
}


class IndexValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class IndexRow:
    component: str
    file_no: str
    start_ts: int
    end_ts: int

    def as_tuple(self) -> tuple[str, str, int, int]:
        return (self.component, self.file_no, self.start_ts, self.end_ts)


def parse_wave_time(value: str) -> int:
    """Parse a UTC timestamp made of whole seconds and return epoch seconds."""
    text = str(value).strip()
    if not text:
        raise ValueError("时刻为空")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"无法解析 UTC 时刻: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    if parsed.microsecond != 0:
        raise ValueError(f"时刻必须为整秒: {value!r}")
    return int(parsed.timestamp())


def format_ts(ts: int) -> str:
    """Format epoch seconds as a UTC ``YYYY-MM-DD HH:MM:SS`` string."""
    return dt.datetime.fromtimestamp(int(ts), tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def day_bounds(day: str) -> tuple[int, int]:
    start = int(
        dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), tzinfo=UTC).timestamp()
    )
    return start, start + DAY_SECONDS


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IndexValidationError(["文件编码无法识别，请使用 UTF-8"])


def _column_map(header: list[str]) -> dict[str, int] | None:
    names = [cell.strip().lower() for cell in header]
    mapping: dict[str, int] = {}
    for key, aliases in (
        ("component", COMP_NAMES),
        ("file_no", FILE_NAMES),
        ("start", START_NAMES),
        ("end", END_NAMES),
    ):
        for index, name in enumerate(names):
            if name in aliases:
                mapping[key] = index
                break
    if len(mapping) != 4:
        return None
    return mapping


def parse_index_csv(data: bytes, day: str) -> list[IndexRow]:
    """Parse and validate a station-day waveform index CSV.

    Returns the canonical rows (possibly empty for a header-only file).
    """
    text = _decode(data)
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise IndexValidationError(["CSV 文件为空"])

    mapping = _column_map(rows[0])
    if mapping is None:
        raise IndexValidationError(
            ["表头无法识别，应包含分量、文件编号、UTC 起始时刻、UTC 结束时刻四列"]
        )
    rows = rows[1:]
    if not rows:
        # Header only: the day's index is cleared on save.
        return []

    day_start, day_end = day_bounds(day)
    errors: list[str] = []
    parsed: list[IndexRow] = []
    seen_pair: set[tuple[str, str]] = set()
    spans_by_file: dict[str, set[tuple[int, int]]] = {}

    for line_no, row in enumerate(rows, start=2):
        if len(row) < 4:
            errors.append(f"第 {line_no} 行缺少列")
            continue
        component = row[mapping["component"]].strip().upper()
        file_no = row[mapping["file_no"]].strip()
        start_text = row[mapping["start"]].strip()
        end_text = row[mapping["end"]].strip()

        if component not in COMPONENTS:
            errors.append(f"第 {line_no} 行分量无效: {component!r}（只允许 Z、N、E）")
            continue
        if not file_no:
            errors.append(f"第 {line_no} 行文件编号为空")
            continue
        pair = (file_no, component)
        if pair in seen_pair:
            errors.append(f"第 {line_no} 行文件编号与分量重复: {file_no} / {component}")
            continue
        try:
            start_ts = parse_wave_time(start_text)
        except ValueError as exc:
            errors.append(f"第 {line_no} 行起始时刻{exc}")
            continue
        try:
            end_ts = parse_wave_time(end_text)
        except ValueError as exc:
            errors.append(f"第 {line_no} 行结束时刻{exc}")
            continue
        if not (day_start <= start_ts < day_end):
            errors.append(
                f"第 {line_no} 行文件必须从所选日 {day}（UTC）开始，实际起始 "
                f"{format_ts(start_ts)}"
            )
            continue
        if end_ts <= start_ts:
            errors.append(f"第 {line_no} 行结束时刻必须晚于起始时刻")
            continue
        if end_ts - start_ts > MAX_FILE_SECONDS:
            errors.append(f"第 {line_no} 行文件时长超过 24 小时")
            continue
        seen_pair.add(pair)
        spans_by_file.setdefault(file_no, set()).add((start_ts, end_ts))
        parsed.append(IndexRow(component, file_no, start_ts, end_ts))

    for file_no, spans in spans_by_file.items():
        if len(spans) > 1:
            errors.append(
                f"同一文件编号 {file_no} 的各分量起止时刻必须一致，"
                f"出现了 {len(spans)} 种不同区间"
            )

    if errors:
        raise IndexValidationError(errors)
    return parsed
