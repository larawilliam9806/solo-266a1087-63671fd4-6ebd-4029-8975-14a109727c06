from __future__ import annotations

import csv
import io
import json
import os
import re
import threading

from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from .catalog import (
    MAX_SPAN_DAYS,
    STATUS_BLOCKED,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PENDING,
    Store,
    parse_day,
    source_from_env,
)
from .collector import collect
from .csv_import import CsvValidationError, parse_csv, verify_includes_collected
from .waveform import IndexValidationError, format_ts, parse_index_csv

STATUS_LABELS = {
    STATUS_PENDING: "待查询",
    STATUS_COMPLETE: "已查全",
    STATUS_BLOCKED: "待补档",
    STATUS_FAILED: "待重试",
}

STATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")


def valid_station(value: str) -> str:
    text = str(value or "").strip()
    if not STATION_RE.fullmatch(text):
        raise ValueError("台站代码需为 1~32 位字母、数字或 . _ -，且以字母数字开头")
    return text


def create_app(
    database: str | None = None,
    source=None,
    testing: bool = False,
) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DATABASE=database or os.environ.get("DATABASE", "/data/archive.db"),
        MAX_CONTENT_LENGTH=4 * 1024 * 1024,
        TESTING=testing,
    )
    store = Store(app.config["DATABASE"])
    store.init()
    app.extensions["store"] = store
    app.extensions["source"] = source or source_from_env()
    work_lock = threading.Lock()

    def current_store() -> Store:
        return app.extensions["store"]

    def current_source():
        return app.extensions["source"]

    @app.template_filter("status_label")
    def status_label(value: str) -> str:
        return STATUS_LABELS.get(value, value)

    @app.template_filter("utc_ts")
    def utc_ts(value: int) -> str:
        return format_ts(int(value))

    @app.template_filter("json_pairs")
    def json_pairs(value: str) -> list[list[int]]:
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return []

    @app.get("/")
    def index():
        st: Store = current_store()
        return render_template(
            "index.html",
            runs=st.runs(),
            max_span=MAX_SPAN_DAYS,
            page_size=current_source().page_size(),
        )

    @app.post("/runs")
    def create_run():
        st: Store = current_store()
        try:
            start = parse_day(request.form.get("start_date", ""))
            end = parse_day(request.form.get("end_date", ""))
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        if end < start:
            flash("结束日期不能早于开始日期", "error")
            return redirect(url_for("index"))
        span = (end - start).days + 1
        if span > MAX_SPAN_DAYS:
            flash(f"一次最多选择 {MAX_SPAN_DAYS} 个连续日", "error")
            return redirect(url_for("index"))
        run_id = st.create_run(start, end)
        flash(f"已创建 {start} 至 {end} 的归档任务", "info")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<int:run_id>")
    def run_detail(run_id: int):
        st: Store = current_store()
        run = st.get_run(run_id)
        if run is None:
            abort(404)
        days = st.days(run_id)
        counts = {status: 0 for status in STATUS_LABELS}
        for row in days:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return render_template(
            "run.html",
            run=run,
            days=days,
            counts=counts,
            labels=STATUS_LABELS,
            ready=st.is_ready(run_id),
            events=st.events(run_id),
            stations=st.stations(),
            lists=st.lists_for_run(run_id),
        )

    @app.post("/runs/<int:run_id>/collect")
    def run_collect(run_id: int):
        st: Store = current_store()
        if st.get_run(run_id) is None:
            abort(404)
        if not work_lock.acquire(blocking=False):
            flash("已有采集或补档操作在进行，请稍后重试", "warn")
            return redirect(url_for("run_detail", run_id=run_id))
        try:
            summary = collect(st, current_source(), run_id)
            flash(
                f"本次采集：查全 {summary.completed} 天，待补档 {summary.blocked} 天，"
                f"待重试 {summary.failed} 天",
                "info" if not summary.failed else "warn",
            )
        finally:
            work_lock.release()
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<int:run_id>/days/<day>/upload")
    def upload_day(run_id: int, day: str):
        st: Store = current_store()
        run = st.get_run(run_id)
        if run is None:
            abort(404)
        rows = [r for r in st.days(run_id) if r["day"] == day]
        if not rows:
            abort(404)
        if rows[0]["status"] != STATUS_BLOCKED:
            flash(f"{day} 不是待补档日期，无需上传", "warn")
            return redirect(url_for("run_detail", run_id=run_id))

        upload = request.files.get("csv")
        if upload is None or not upload.filename:
            flash("请选择 CSV 文件", "error")
            return redirect(url_for("run_detail", run_id=run_id))

        try:
            parsed = parse_csv(upload.read(), day)
            st.backfill_day(
                run_id,
                day,
                parsed.events,
                verify=lambda existing: verify_includes_collected(parsed, existing),
            )
        except CsvValidationError as exc:
            flash("补档文件校验失败，进度保持不变：" + "；".join(exc.errors), "error")
            return redirect(url_for("run_detail", run_id=run_id))
        except ValueError as exc:
            flash(f"补档失败，进度保持不变：{exc}", "error")
            return redirect(url_for("run_detail", run_id=run_id))

        flash(f"{day} 已补档，共 {len(parsed.events)} 条", "info")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<int:run_id>/download")
    def download_run(run_id: int):
        st: Store = current_store()
        run = st.get_run(run_id)
        if run is None:
            abort(404)
        if not st.is_ready(run_id):
            flash("还有未查全或未补档的日期，暂不能下载完整目录", "error")
            return redirect(url_for("run_detail", run_id=run_id))
        events = st.events(run_id)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["event_id", "utc_time", "magnitude"])
        for row in events:
            writer.writerow([row["event_id"], row["utc_time"], f"{row['magnitude']:g}"])
        filename = f"catalog_{run['start_date']}_{run['end_date']}.csv"
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    # ------------------------------------------------------------------
    # Waveform indexes
    # ------------------------------------------------------------------

    @app.get("/stations")
    def stations():
        st: Store = current_store()
        names = st.stations()
        summary = {name: st.index_days(name) for name in names}
        return render_template("stations.html", stations=names, summary=summary)

    @app.post("/stations")
    def create_station():
        try:
            station = valid_station(request.form.get("station", ""))
            day = parse_day(request.form.get("day", "")).isoformat()
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("stations"))
        st: Store = current_store()
        if station in st.stations():
            flash(f"台站 {station} 已存在，请直接选择该台站上传索引", "warn")
            return redirect(url_for("station_detail", station=station))
        # A station only materialises with its first index row; direct the user
        # to its detail page where the first-day CSV can be uploaded.
        flash(f"新台站 {station}：请在该台站页面选择 {day} 的波形索引 CSV 并保存", "info")
        return redirect(url_for("station_detail", station=station))

    @app.get("/stations/<station>")
    def station_detail(station: str):
        st: Store = current_store()
        if not STATION_RE.fullmatch(station):
            abort(404)
        known = set(st.stations())
        days = st.index_days(station) if station in known else []
        return render_template(
            "station.html",
            station=station,
            days=days,
            exists=station in known,
        )

    @app.get("/stations/<station>/days/<day>")
    def station_day(station: str, day: str):
        st: Store = current_store()
        try:
            day = parse_day(day).isoformat()
        except ValueError:
            abort(404)
        if not STATION_RE.fullmatch(station):
            abort(404)
        rows = st.index_rows(station, day)
        return render_template(
            "station_day.html", station=station, day=day, rows=rows
        )

    @app.post("/stations/<station>/index")
    def upload_index_form(station: str):
        try:
            day = parse_day(request.form.get("day", "")).isoformat()
        except ValueError:
            flash("日期必须为 YYYY-MM-DD", "error")
            return redirect(url_for("station_detail", station=station))
        return _save_index(station, day, request.files.get("csv"))

    @app.post("/stations/<station>/days/<day>/index")
    def upload_index(station: str, day: str):
        try:
            station = valid_station(station)
            day = parse_day(day).isoformat()
        except ValueError:
            abort(404)
        return _save_index(station, day, request.files.get("csv"))

    def _save_index(station: str, day: str, upload):
        st: Store = current_store()
        try:
            station = valid_station(station)
        except ValueError:
            abort(404)
        if upload is None or not upload.filename:
            flash("请选择波形索引 CSV 文件", "error")
            return redirect(url_for("station_detail", station=station))
        try:
            rows = parse_index_csv(upload.read(), day)
            result = st.replace_index(station, day, rows)
        except IndexValidationError as exc:
            flash("索引校验失败，索引与清单保持不变：" + "；".join(exc.errors), "error")
            return redirect(url_for("station_detail", station=station))
        except ValueError as exc:
            flash(f"索引保存失败，索引与清单保持不变：{exc}", "error")
            return redirect(url_for("station_detail", station=station))
        if not rows:
            flash(
                f"{station} 在 {day} 的索引已清空"
                + (
                    f"，并重算 {result['windows_recomputed']} 条相交窗口"
                    if result["windows_recomputed"]
                    else ""
                ),
                "info",
            )
        else:
            flash(
                f"{station} 在 {day} 的索引已保存（{len(rows)} 个分量段，"
                f"重算 {result['windows_recomputed']} 条相交窗口）",
                "info",
            )
        return redirect(url_for("station_day", station=station, day=day))

    # ------------------------------------------------------------------
    # Retrieval lists
    # ------------------------------------------------------------------

    @app.post("/runs/<int:run_id>/lists")
    def create_list(run_id: int):
        st: Store = current_store()
        if st.get_run(run_id) is None:
            abort(404)
        try:
            station = valid_station(request.form.get("station", ""))
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("run_detail", run_id=run_id))
        list_id = st.generate_retrieval_list(run_id, station)
        flash(f"已生成台站 {station} 的调阅清单", "info")
        return redirect(url_for("list_detail", list_id=list_id))

    @app.get("/lists/<int:list_id>")
    def list_detail(list_id: int):
        st: Store = current_store()
        header = st.get_list(list_id)
        if header is None:
            abort(404)
        windows = st.list_windows(list_id)
        events: dict[str, dict] = {}
        for row in windows:
            bucket = events.setdefault(
                row["event_id"],
                {
                    "event_id": row["event_id"],
                    "event_time": row["event_time"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "components": {},
                },
            )
            bucket["components"][row["component"]] = row
        ordered = [
            events[key]
            for key in sorted(events, key=lambda k: (events[k]["event_time"], k))
        ]
        all_covered = bool(windows) and all(row["covered"] for row in windows)
        return render_template(
            "list.html",
            header=header,
            events=ordered,
            all_covered=all_covered,
            window_rows=len(windows),
        )

    @app.post("/lists/<int:list_id>/enable")
    def enable_list(list_id: int):
        st: Store = current_store()
        if st.get_list(list_id) is None:
            abort(404)
        st.set_list_enabled(list_id, True)
        flash("清单已启用；后续日期查全时，事件与调阅窗口将在同一事务中生效", "info")
        return redirect(url_for("list_detail", list_id=list_id))

    @app.post("/lists/<int:list_id>/disable")
    def disable_list(list_id: int):
        st: Store = current_store()
        if st.get_list(list_id) is None:
            abort(404)
        st.set_list_enabled(list_id, False)
        flash("清单已停用，保留当前内容，不再随采集自动更新", "info")
        return redirect(url_for("list_detail", list_id=list_id))

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("404.html"), 404

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        flash("上传文件过大（上限 4 MB）", "error")
        return redirect(request.referrer or url_for("index"))

    return app
