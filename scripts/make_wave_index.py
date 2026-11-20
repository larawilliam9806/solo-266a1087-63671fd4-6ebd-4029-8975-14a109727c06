#!/usr/bin/env python3
"""Generate demo waveform-index CSVs for a station and UTC day.

The files follow the upload contract exactly: header
``component,file_no,start,end``, one row per component, whole-second
half-open UTC intervals, files starting on the selected day and lasting at
most 24 hours.  Two layouts are produced for the same day so the retrieval
list can show both full coverage and a coverage gap:

* ``wave_<station>_<day>_full.csv``  -- one 24 h file per component;
* ``wave_<station>_<day>_gap.csv``   -- the Z component loses a chunk that
  intersects the demo event windows, N/E stay fully covered.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

COMPONENTS = ("Z", "N", "E")


def write_rows(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["component", "file_no", "start", "end"])
        writer.writerows(rows)


def iso(day: dt.date, seconds: int) -> str:
    stamp = dt.datetime.combine(day, dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(
        seconds=seconds
    )
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("station")
    parser.add_argument("day", help="UTC date YYYY-MM-DD")
    parser.add_argument("--out-dir", default="seed")
    args = parser.parse_args()

    day = dt.date.fromisoformat(args.day)
    out = Path(args.out_dir)

    full = [
        (comp, "W0001", iso(day, 0), iso(day, 86400))
        for comp in COMPONENTS
    ]
    full_path = out / f"wave_{args.station}_{args.day}_full.csv"
    write_rows(full_path, full)

    # Z: two files with a 10-minute hole around 03:00 local event time.
    gap_rows = [
        ("Z", "W0002", iso(day, 0), iso(day, 3 * 3600)),
        ("Z", "W0003", iso(day, 3 * 3600 + 600), iso(day, 86400)),
        ("N", "W0001", iso(day, 0), iso(day, 86400)),
        ("E", "W0001", iso(day, 0), iso(day, 86400)),
    ]
    gap_path = out / f"wave_{args.station}_{args.day}_gap.csv"
    write_rows(gap_path, gap_rows)

    print(f"wrote {full_path} and {gap_path}")


if __name__ == "__main__":
    main()
