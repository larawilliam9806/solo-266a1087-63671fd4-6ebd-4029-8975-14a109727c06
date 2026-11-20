import datetime as dt
import io
import json

import pytest

from archive_app import create_app
from archive_app.catalog import (
    LIST_ENABLED,
    Event,
    MockCatalogSource,
    Store,
)
from archive_app.planner import Segment, event_window, plan_component, plan_event
from archive_app.waveform import (
    COMPONENTS,
    IndexValidationError,
    day_bounds,
    format_ts,
    parse_index_csv,
    parse_wave_time,
)


def make_store(tmp_path):
    store = Store(str(tmp_path / "wave.db"))
    store.init()
    return store


def ts(day, hh=0, mm=0, ss=0):
    return int(
        dt.datetime.combine(
            dt.date.fromisoformat(day),
            dt.time(hh, mm, ss),
            tzinfo=dt.timezone.utc,
        ).timestamp()
    )


# ---------------------------------------------------------------------------
# Index CSV validation
# ---------------------------------------------------------------------------


def test_index_csv_parses_three_components_and_canonical_times():
    data = (
        "component,file_no,start,end\n"
        "Z,F1,2024-03-01T00:00:00Z,2024-03-01T12:00:00Z\n"
        "N,F1,2024-03-01T00:00:00Z,2024-03-01T12:00:00Z\n"
        "E,F1,2024-03-01 00:00:00,2024-03-01 12:00:00\n"
    ).encode()
    rows = parse_index_csv(data, "2024-03-01")
    assert [r.component for r in rows] == ["Z", "N", "E"]
    assert {r.file_no for r in rows} == {"F1"}
    assert rows[0].start_ts == ts("2024-03-01")
    assert rows[0].end_ts == ts("2024-03-01", 12)


def test_index_csv_allows_cross_midnight_up_to_24h():
    data = (
        "component,file_no,start,end\n"
        "Z,F1,2024-03-01T23:00:00Z,2024-03-02T01:00:00Z\n"
    ).encode()
    rows = parse_index_csv(data, "2024-03-01")
    assert len(rows) == 1
    assert rows[0].end_ts - rows[0].start_ts == 2 * 3600


def test_index_header_only_clears_day():
    assert parse_index_csv(b"component,file_no,start,end\n", "2024-03-01") == []


def test_index_rejects_bad_component_duplicate_pair_and_blank_file():
    data = (
        "component,file_no,start,end\n"
        "X,F1,2024-03-01T00:00:00Z,2024-03-01T01:00:00Z\n"
        "Z,F1,2024-03-01T00:00:00Z,2024-03-01T01:00:00Z\n"
        "Z,F1,2024-03-01T02:00:00Z,2024-03-01T03:00:00Z\n"
        "Z,,2024-03-01T02:00:00Z,2024-03-01T03:00:00Z\n"
    ).encode()
    with pytest.raises(IndexValidationError) as exc:
        parse_index_csv(data, "2024-03-01")
    text = " ".join(exc.value.errors)
    assert "分量" in text and "重复" in text and "文件编号为空" in text


def test_index_rejects_wrong_start_day_non_integer_second_and_over_24h():
    data = (
        "component,file_no,start,end\n"
        "Z,F1,2024-02-29T23:00:00Z,2024-03-01T01:00:00Z\n"
        "Z,F2,2024-03-01T00:00:00.5Z,2024-03-01T01:00:00Z\n"
        "Z,F3,2024-03-01T00:00:00Z,2024-03-02T00:00:01Z\n"
        "Z,F4,2024-03-01T02:00:00Z,2024-03-01T01:00:00Z\n"
    ).encode()
    with pytest.raises(IndexValidationError) as exc:
        parse_index_csv(data, "2024-03-01")
    text = " ".join(exc.value.errors)
    assert "所选日" in text
    assert "整秒" in text
    assert "24 小时" in text
    assert "晚于" in text


def test_index_rejects_inconsistent_same_file_spans():
    data = (
        "component,file_no,start,end\n"
        "Z,F1,2024-03-01T00:00:00Z,2024-03-01T12:00:00Z\n"
        "N,F1,2024-03-01T00:00:00Z,2024-03-01T13:00:00Z\n"
    ).encode()
    with pytest.raises(IndexValidationError) as exc:
        parse_index_csv(data, "2024-03-01")
    assert "起止时刻必须一致" in " ".join(exc.value.errors)


def test_index_rejects_unknown_header_and_empty_file():
    with pytest.raises(IndexValidationError):
        parse_index_csv(b"a,b,c,d\n", "2024-03-01")
    with pytest.raises(IndexValidationError):
        parse_index_csv(b"", "2024-03-01")


def test_day_bounds_and_helpers():
    start, end = day_bounds("2024-03-01")
    assert end - start == 86400
    assert format_ts(start) == "2024-03-01 00:00:00"
    assert parse_wave_time("2024-03-01T00:00:00Z") == start


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def origin_ts(day="2024-03-01", hh=3):
    return ts(day, hh)


def test_window_is_half_open_30_before_90_after():
    ws, we = event_window(origin_ts())
    assert we - ws == 120
    assert origin_ts() - ws == 30
    assert we - origin_ts() == 90


def test_planner_picks_minimum_files_with_non_overlapping_cuts():
    o = origin_ts()
    ws, we = event_window(o)
    segments = [
        Segment("FA", ws - 100, o),
        Segment("FB", ws - 10, we + 100),  # alone covers the whole window
    ]
    plan = plan_component("Z", ws, we, segments)
    assert plan.covered
    assert [c.file_no for c in plan.cuts] == ["FB"]
    assert plan.cuts[0].start_ts == ws and plan.cuts[0].end_ts == we
    # cuts within one component never overlap
    for earlier, later in zip(plan.cuts, plan.cuts[1:]):
        assert earlier.end_ts <= later.start_ts


def test_planner_stitches_two_files():
    o = origin_ts()
    ws, we = event_window(o)
    segments = [
        Segment("FA", ws - 10, o),
        Segment("FB", o, we + 10),
    ]
    plan = plan_component("Z", ws, we, segments)
    assert plan.covered
    assert [c.file_no for c in plan.cuts] == ["FA", "FB"]
    assert plan.cuts[0].end_ts == plan.cuts[1].start_ts == o


def test_planner_reports_gaps_for_middle_hole_and_missing_component():
    o = origin_ts()
    ws, we = event_window(o)
    segments = [Segment("FA", ws - 5, o), Segment("FB", o + 30, we + 5)]
    plan = plan_component("Z", ws, we, segments)
    assert not plan.covered
    assert plan.gaps == [(o, o + 30)]
    assert [c.file_no for c in plan.cuts] == ["FA", "FB"]

    empty = plan_component("N", ws, we, [])
    assert empty.gaps == [(ws, we)] and empty.cuts == []


def test_planner_deterministic_tie_break():
    ws, we = event_window(origin_ts())
    segments = [
        Segment("FZ", ws - 1, we + 1),
        Segment("FA", ws - 1, we + 1),
    ]
    plan = plan_component("Z", ws, we, segments)
    assert plan.cuts[0].file_no == "FA"


def test_plan_event_covers_all_three_components():
    ws, we = event_window(origin_ts())
    segments = {c: [Segment("F1", ws - 5, we + 5)] for c in COMPONENTS}
    plans = plan_event(origin_ts(), segments)
    assert set(plans) == set(COMPONENTS)
    assert all(p.covered for p in plans.values())


# ---------------------------------------------------------------------------
# Store: index replacement, file-number uniqueness, incremental recompute
# ---------------------------------------------------------------------------


def full_day_index(day, file_no, components=COMPONENTS):
    from archive_app.waveform import IndexRow

    start, end = day_bounds(day)
    return [IndexRow(c, file_no, start, end) for c in components]


def test_replace_index_persists_and_header_only_clears(tmp_path):
    store = make_store(tmp_path)
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    assert store.stations() == ["S1"]
    assert len(store.index_rows("S1", "2024-03-01")) == 3
    store.replace_index("S1", "2024-03-01", [])
    assert store.stations() == []
    assert store.index_rows("S1", "2024-03-01") == []


def test_file_number_unique_within_station_across_days(tmp_path):
    store = make_store(tmp_path)
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    # Same number on another day of the same station is rejected, but the same
    # number may be used at a different station.
    with pytest.raises(ValueError, match="文件编号"):
        store.replace_index("S1", "2024-03-02", full_day_index("2024-03-02", "F1"))
    store.replace_index("S2", "2024-03-02", full_day_index("2024-03-02", "F1"))
    assert [r["day"] for r in store.index_days("S2")] == ["2024-03-02"]
    assert len(store.index_rows("S1", "2024-03-01")) == 3


def test_failed_replacement_preserves_prior_index(tmp_path):
    store = make_store(tmp_path)
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    before = [dict(r) for r in store.index_rows("S1", "2024-03-01")]
    with pytest.raises(ValueError):
        store.replace_index("S1", "2024-03-02", full_day_index("2024-03-02", "F1"))
    assert [dict(r) for r in store.index_rows("S1", "2024-03-01")] == before


def _seed_run_with_event(store, run_id=1, day="2024-03-01", event_id="E1"):
    store.create_run(dt.date.fromisoformat(day), dt.date.fromisoformat(day))
    store.complete_days(
        run_id,
        {day: [Event(event_id, f"{day} 03:00:00", 2.0)]},
    )


def test_incremental_recompute_only_touches_intersecting_windows(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(dt.date(2024, 3, 1), dt.date(2024, 3, 10))
    store.complete_days(
        run_id,
        {
            "2024-03-01": [Event("E1", "2024-03-01 03:00:00", 2.0)],
            "2024-03-10": [Event("E2", "2024-03-10 03:00:00", 2.0)],
        },
    )
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    list_id = store.generate_retrieval_list(run_id, "S1")
    windows_before = {r["event_id"]: dict(r) for r in store.list_windows(list_id)}
    assert windows_before["E1"]["covered"]
    assert not windows_before["E2"]["covered"]  # no index on 2024-03-10 yet
    remote_before = windows_before["E2"]

    # Replace the day-1 index with a file that misses the E1 window.  Only the
    # three E1 windows intersect the swapped files; E2 stays byte-identical.
    from archive_app.waveform import IndexRow

    short = [
        IndexRow(c, "F9", ts("2024-03-01", 10), ts("2024-03-01", 11))
        for c in COMPONENTS
    ]
    result = store.replace_index("S1", "2024-03-01", short)
    assert result["windows_recomputed"] == 3
    after = {
        (r["event_id"], r["component"]): dict(r)
        for r in store.list_windows(list_id)
    }
    assert all(after[("E1", c)]["covered"] == 0 for c in COMPONENTS)
    ws, we = event_window(origin_ts())
    assert json.loads(after[("E1", "Z")]["gaps"]) == [[ws, we]]
    assert dict(after[("E2", "Z")]) == remote_before


def test_index_change_is_scoped_to_one_station(tmp_path):
    store = make_store(tmp_path)
    _seed_run_with_event(store)
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    store.replace_index("S2", "2024-03-01", full_day_index("2024-03-01", "G1"))
    l1 = store.generate_retrieval_list(1, "S1")
    l2 = store.generate_retrieval_list(1, "S2")
    snapshot = [dict(r) for r in store.list_windows(l1)]

    store.replace_index("S2", "2024-03-01", full_day_index("2024-03-01", "G2"))
    assert [dict(r) for r in store.list_windows(l1)] == snapshot
    changed = [dict(r) for r in store.list_windows(l2)]
    assert all("G2" in r["cuts"] for r in changed)


def test_enabled_list_syncs_inside_completion_transaction(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(dt.date(2024, 3, 1), dt.date(2024, 3, 2))
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    list_id = store.generate_retrieval_list(run_id, "S1")
    assert store.list_windows(list_id) == []
    store.set_list_enabled(list_id, True)

    store.complete_days(
        run_id, {"2024-03-01": [Event("E1", "2024-03-01 03:00:00", 2.0)]}
    )
    windows = store.list_windows(list_id)
    assert {r["event_id"] for r in windows} == {"E1"}
    assert all(r["covered"] for r in windows)
    assert store.get_list(list_id)["enabled"] == LIST_ENABLED

    # A failing completion rolls back: neither progress nor list changes.
    before = [dict(r) for r in windows]
    with pytest.raises(ValueError):
        # Duplicate event id violates the primary key -> whole tx rolls back.
        store.complete_days(
            run_id, {"2024-03-02": [Event("E1", "2024-03-02 03:00:00", 2.0)]}
        )
    assert [dict(r) for r in store.list_windows(list_id)] == before
    statuses = {r["day"]: r["status"] for r in store.days(run_id)}
    assert statuses["2024-03-02"] == "pending"


def test_disabled_list_does_not_sync_on_completion(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(dt.date(2024, 3, 1), dt.date(2024, 3, 2))
    store.replace_index("S1", "2024-03-01", full_day_index("2024-03-01", "F1"))
    store.complete_days(
        run_id, {"2024-03-01": [Event("E1", "2024-03-01 03:00:00", 2.0)]}
    )
    list_id = store.generate_retrieval_list(run_id, "S1")  # disabled by default
    assert store.get_list(list_id)["enabled"] == 0
    assert {r["event_id"] for r in store.list_windows(list_id)} == {"E1"}

    # Completing another day while the list is disabled never adds windows.
    store.complete_days(
        run_id, {"2024-03-02": [Event("E2", "2024-03-02 03:00:00", 2.0)]}
    )
    assert "E2" not in {r["event_id"] for r in store.list_windows(list_id)}

    # Regenerating (or enabling) picks up the now-complete second day.
    store.set_list_enabled(list_id, True)
    assert {r["event_id"] for r in store.list_windows(list_id)} == {"E1", "E2"}


# ---------------------------------------------------------------------------
# Flask flows
# ---------------------------------------------------------------------------


def write_catalog(tmp_path, events):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return str(path)


def make_client(tmp_path, events):
    path = write_catalog(tmp_path, events)
    app = create_app(
        database=str(tmp_path / "web.db"),
        source=MockCatalogSource(path),
        testing=True,
    )
    return app.test_client(), app


def upload_index(client, station, day, content, filename="index.csv"):
    return client.post(
        f"/stations/{station}/days/{day}/index",
        data={"csv": (io.BytesIO(content.encode("utf-8")), filename)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def full_csv(day, file_no, end_day=None):
    if end_day is None:
        end_day = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
    lines = ["component,file_no,start,end"]
    for comp in COMPONENTS:
        lines.append(f"{comp},{file_no},{day}T00:00:00Z,{end_day}T00:00:00Z")
    return "\n".join(lines) + "\n"


def test_web_full_retrieval_flow_with_gap_and_recompute(tmp_path):
    events = [{"id": "E1", "time": "2024-03-01T03:00:00Z", "magnitude": 2.0}]
    client, app = make_client(tmp_path, events)
    store = app.extensions["store"]

    client.post("/runs", data={"start_date": "2024-03-01", "end_date": "2024-03-01"})
    client.post("/runs/1/collect")

    resp = upload_index(client, "STA01", "2024-03-01", full_csv("2024-03-01", "F1"))
    assert resp.status_code == 200
    page = client.get("/stations").get_data(as_text=True)
    assert "STA01" in page

    resp = client.post("/runs/1/lists", data={"station": "STA01"}, follow_redirects=True)
    page = resp.get_data(as_text=True)
    assert "调阅清单" in page and "连续覆盖" in page and "F1" in page

    # Shrink the Z component -> gap appears and only intersecting windows update.
    gappy = (
        "component,file_no,start,end\n"
        "Z,F9,2024-03-01T10:00:00Z,2024-03-01T11:00:00Z\n"
        "N,F1,2024-03-01T00:00:00Z,2024-03-02T00:00:00Z\n"
        "E,F1,2024-03-01T00:00:00Z,2024-03-02T00:00:00Z\n"
    )
    upload_index(client, "STA01", "2024-03-01", gappy)
    page = client.get("/lists/1").get_data(as_text=True)
    assert "覆盖不足" in page and "缺" in page
    z = [dict(r) for r in store.list_windows(1) if r["component"] == "Z"][0]
    assert z["covered"] == 0 and json.loads(z["gaps"])

    # Enable, then upload a header-only CSV: index clears, list keeps windows
    # but every component becomes uncovered.
    client.post("/lists/1/enable")
    resp = upload_index(client, "STA01", "2024-03-01", "component,file_no,start,end\n")
    assert "已清空" in resp.get_data(as_text=True)
    assert store.stations() == []
    assert all(dict(r)["covered"] == 0 for r in store.list_windows(1))


def test_web_invalid_index_keeps_index_and_list_unchanged(tmp_path):
    events = [{"id": "E1", "time": "2024-03-01T03:00:00Z", "magnitude": 2.0}]
    client, app = make_client(tmp_path, events)
    store = app.extensions["store"]
    client.post("/runs", data={"start_date": "2024-03-01", "end_date": "2024-03-01"})
    client.post("/runs/1/collect")
    upload_index(client, "STA01", "2024-03-01", full_csv("2024-03-01", "F1"))
    client.post("/runs/1/lists", data={"station": "STA01"})
    index_before = [dict(r) for r in store.index_rows("STA01", "2024-03-01")]
    list_before = [dict(r) for r in store.list_windows(1)]

    bad = "component,file_no,start,end\nZ,F2,2024-03-01T00:00:00Z,2024-03-03T00:00:00Z\n"
    resp = upload_index(client, "STA01", "2024-03-01", bad)
    assert "校验失败" in resp.get_data(as_text=True)
    assert [dict(r) for r in store.index_rows("STA01", "2024-03-01")] == index_before
    assert [dict(r) for r in store.list_windows(1)] == list_before


def test_web_enabled_list_takes_effect_with_backfill_transaction(tmp_path):
    # Hours 1..22 keep every [origin-30, origin+90) window inside the UTC day.
    events = [{"id": f"X{i:02d}", "time": f"2024-03-01T{i:02d}:00:00Z", "magnitude": 2.0}
              for i in range(1, 23)]
    client, app = make_client(tmp_path, events)
    store = app.extensions["store"]
    client.post("/runs", data={"start_date": "2024-03-01", "end_date": "2024-03-01"})
    client.post("/runs/1/collect")
    assert {r["day"]: r["status"] for r in store.days(1)}["2024-03-01"] == "blocked"

    upload_index(client, "STA01", "2024-03-01", full_csv("2024-03-01", "F1"))
    list_id = store.generate_retrieval_list(1, "STA01")
    store.set_list_enabled(list_id, True)
    assert store.list_windows(list_id) == []  # blocked day contributes no events

    rows = "event_id,utc_time,magnitude\n"
    for i in range(1, 23):
        rows += f"X{i:02d},2024-03-01 {i:02d}:00:00,2.0\n"
    resp = client.post(
        "/runs/1/days/2024-03-01/upload",
        data={"csv": (io.BytesIO(rows.encode()), "b.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    windows = store.list_windows(list_id)
    assert len({r["event_id"] for r in windows}) == 22
    assert len(windows) == 22 * 3
    assert all(r["covered"] for r in windows)


def test_web_list_page_reads_saved_content_only(tmp_path):
    events = [{"id": "E1", "time": "2024-03-01T03:00:00Z", "magnitude": 2.0}]
    client, app = make_client(tmp_path, events)
    store = app.extensions["store"]
    client.post("/runs", data={"start_date": "2024-03-01", "end_date": "2024-03-01"})
    client.post("/runs/1/collect")
    upload_index(client, "STA01", "2024-03-01", full_csv("2024-03-01", "F1"))
    client.post("/runs/1/lists", data={"station": "STA01"})

    # Tamper with the index afterwards without triggering recompute simulation:
    # the page must still render the persisted rows (no live replanning).
    saved = [dict(r) for r in store.list_windows(1)]
    page = client.get("/lists/1").get_data(as_text=True)
    assert "2024-03-01 02:59:30" in page and "2024-03-01 03:01:30" in page
    assert [dict(r) for r in store.list_windows(1)] == saved


def test_web_station_day_detail_and_unknown_station_404(tmp_path):
    client, _ = make_client(tmp_path, [])
    resp = client.get("/stations/BAD%21NAME")
    assert resp.status_code == 404
    # A syntactically valid, not-yet-seen station page is reachable for first upload.
    assert client.get("/stations/NEW1").status_code == 200
    assert client.get("/stations/NEW1/days/2024-03-01").status_code == 200
