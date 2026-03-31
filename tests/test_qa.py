import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from app.core.alert_history import AlertHistoryStore
from app.core.qa import (
    AnswerResult,
    QAService,
    _extract_image_path,
    _utc_results_to_ist,
)


def _make_db(tmp_path, alerts=None):
    db_path = str(tmp_path / "alerts.db")
    store = AlertHistoryStore(db_path)
    for alert in alerts or []:
        store.insert_alert(**alert)
    return store, db_path


# ---------------------------------------------------------------------------
# Fixture: real data captured 2026-03-30 (IST day = UTC 2026-03-29T18:30 .. 2026-03-30T18:30)
#   5 dog, 1 cat, 49 person, 7 motorcycle-in-context
# ---------------------------------------------------------------------------

def _make_real_day_alerts():
    """Representative subset from 2026-03-30 production data."""
    return [
        dict(ts=datetime(2026, 3, 29, 18, 53, 25, tzinfo=timezone.utc).timestamp(),
             count=1, best_conf=0.704, image_path="/workspace/work/alerts/snap_dog1.jpg",
             trigger_classes=["dog"], context_classes=["dog"]),
        dict(ts=datetime(2026, 3, 29, 18, 55, 55, tzinfo=timezone.utc).timestamp(),
             count=1, best_conf=0.672, image_path="/workspace/work/alerts/snap_dog2.jpg",
             trigger_classes=["dog"], context_classes=["dog"]),
        dict(ts=datetime(2026, 3, 29, 18, 58, 29, tzinfo=timezone.utc).timestamp(),
             count=1, best_conf=0.638, image_path="/workspace/work/alerts/snap_cat1.jpg",
             trigger_classes=["cat"], context_classes=["cat"]),
        dict(ts=datetime(2026, 3, 29, 23, 51, 30, tzinfo=timezone.utc).timestamp(),
             count=1, best_conf=0.908, image_path="/workspace/work/alerts/snap_person1.jpg",
             trigger_classes=["person"], context_classes=["person"]),
        dict(ts=datetime(2026, 3, 30, 2, 30, 38, tzinfo=timezone.utc).timestamp(),
             count=1, best_conf=0.836, image_path="/workspace/work/alerts/snap_person2.jpg",
             trigger_classes=["person"], context_classes=["motorcycle", "person"]),
        dict(ts=datetime(2026, 3, 30, 7, 48, 50, tzinfo=timezone.utc).timestamp(),
             count=1, best_conf=0.764, image_path="/workspace/work/alerts/snap_person3.jpg",
             trigger_classes=["person"], context_classes=["person"]),
    ]


# ---------------------------------------------------------------------------
# Timestamp storage format
# ---------------------------------------------------------------------------

def test_timestamps_stored_without_offset(tmp_path):
    """Stored timestamps must be bare UTC (no +00:00) for reliable text comparison."""
    store, db_path = _make_db(tmp_path, _make_real_day_alerts())
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT ts FROM alerts").fetchall()
    conn.close()
    for (ts,) in rows:
        assert "+00:00" not in ts, f"Timestamp still has offset: {ts}"
        assert "T" in ts, f"Unexpected format: {ts}"


def test_legacy_offset_migrated_on_init(tmp_path):
    """Opening the DB should strip +00:00 from any legacy rows."""
    db_path = str(tmp_path / "alerts.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        count INTEGER NOT NULL, best_conf REAL NOT NULL,
        image_path TEXT,
        trigger_classes TEXT NOT NULL DEFAULT '[]',
        context_classes TEXT NOT NULL DEFAULT '[]')""")
    conn.execute("INSERT INTO alerts(ts, count, best_conf) VALUES ('2026-03-29T18:53:25+00:00', 1, 0.9)")
    conn.commit()
    conn.close()
    AlertHistoryStore(db_path)
    conn = sqlite3.connect(db_path)
    ts = conn.execute("SELECT ts FROM alerts").fetchone()[0]
    conn.close()
    assert ts == "2026-03-29T18:53:25"


# ---------------------------------------------------------------------------
# UTC → IST conversion for answer display
# ---------------------------------------------------------------------------

def test_utc_to_ist_basic():
    assert "2026-03-30 12:23 AM IST" in _utc_results_to_ist("2026-03-29T18:53:25")


def test_utc_to_ist_with_fractional_seconds():
    """Fractional seconds should be consumed, not left dangling."""
    result = _utc_results_to_ist("2026-03-29T18:53:25.066414")
    assert ".066414" not in result
    assert "12:23 AM IST" in result


def test_utc_to_ist_no_false_match():
    """Non-timestamp text should pass through unchanged."""
    text = "count: 5, id: 743"
    assert _utc_results_to_ist(text) == text


# ---------------------------------------------------------------------------
# Image path extraction
# ---------------------------------------------------------------------------

def test_extract_image_path_jpg():
    text = "Here is the result: /workspace/work/alerts/snap_dog1.jpg found."
    assert _extract_image_path(text) == "/workspace/work/alerts/snap_dog1.jpg"


def test_extract_image_path_png():
    text = "Image at /tmp/test.png"
    assert _extract_image_path(text) == "/tmp/test.png"


def test_extract_image_path_none():
    assert _extract_image_path("no image here, count=5") is None


# ---------------------------------------------------------------------------
# SQL views created by AlertHistoryStore
# ---------------------------------------------------------------------------

def test_views_created_on_init(tmp_path):
    """init_db should create v_daily_stats and v_hourly_stats views."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    conn = sqlite3.connect(db_path)
    views = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()]
    conn.close()
    assert "v_daily_stats" in views
    assert "v_hourly_stats" in views


def test_v_daily_stats_returns_ist_dates(tmp_path):
    """v_daily_stats should group by IST date, not UTC date."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT ist_date, total_detections FROM v_daily_stats ORDER BY ist_date"
    ).fetchall()
    conn.close()
    dates = [r[0] for r in rows]
    totals = {r[0]: r[1] for r in rows}
    assert "2026-03-30" in dates
    assert totals["2026-03-30"] == 6  # all 6 alerts fall on Mar 30 IST


def test_v_hourly_stats_returns_ist_hours(tmp_path):
    """v_hourly_stats should use IST hours."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT ist_hour, total_detections FROM v_hourly_stats ORDER BY total_detections DESC"
    ).fetchall()
    conn.close()
    top_hour = rows[0][0]
    assert top_hour == 0  # 2026-03-29T18:53 UTC = 00:23 IST → hour 0


# ---------------------------------------------------------------------------
# QAService basics (no LLM)
# ---------------------------------------------------------------------------

def test_no_llm_returns_config_message(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None)
    answer = svc.answer_question("how many alerts today?")
    assert "LLM is not configured" in answer.text


def test_empty_question_returns_message(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None)
    answer = svc.answer_question("   ")
    assert "Please provide a question" in answer.text


# ---------------------------------------------------------------------------
# QAService with mocked LLM agent
# ---------------------------------------------------------------------------

def _svc_with_mock_agent(tmp_path, alerts=None):
    """Build a QAService with llm=None then inject a mock agent."""
    _, db_path = _make_db(tmp_path, alerts or _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    svc._agent = MagicMock()
    return svc


def test_answer_question_returns_answer_result(tmp_path):
    """answer_question should return an AnswerResult with text."""
    svc = _svc_with_mock_agent(tmp_path)

    from langchain_core.messages import AIMessage
    svc._agent.invoke.return_value = {
        "messages": [AIMessage(content="There were 6 alerts today.")]
    }

    answer = svc.answer_question("how many alerts?")
    assert isinstance(answer, AnswerResult)
    assert "6 alerts" in answer.text


def test_answer_question_extracts_image(tmp_path):
    """answer_question should extract image_path from agent messages."""
    svc = _svc_with_mock_agent(tmp_path)

    from langchain_core.messages import AIMessage, ToolMessage
    svc._agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{"id": "1", "name": "sql_db_query", "args": {"query": "SELECT image_path FROM alerts LIMIT 1"}}]),
            ToolMessage(content="[('/workspace/work/alerts/snap_dog1.jpg',)]", tool_call_id="1"),
            AIMessage(content="Here's the screenshot: /workspace/work/alerts/snap_dog1.jpg"),
        ]
    }

    answer = svc.answer_question("show me a dog pic")
    assert answer.image_path == "/workspace/work/alerts/snap_dog1.jpg"


def test_answer_question_handles_exception(tmp_path):
    """If the agent throws, answer_question should return a friendly error."""
    svc = _svc_with_mock_agent(tmp_path)
    svc._agent.invoke.side_effect = RuntimeError("boom")

    answer = svc.answer_question("test")
    assert "went wrong" in answer.text.lower()


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------

def test_system_prompt_contains_class_names(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None, class_names=("dog", "person"))
    prompt = svc._build_system_prompt()
    assert "dog" in prompt
    assert "person" in prompt


def test_system_prompt_contains_today_boundary(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None)
    prompt = svc._build_system_prompt()
    assert "today_utc_start" not in prompt  # should be formatted, not a placeholder
    assert "T18:30:00" in prompt  # IST midnight = 18:30 UTC


def test_system_prompt_contains_views(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None)
    prompt = svc._build_system_prompt()
    assert "v_daily_stats" in prompt
    assert "v_hourly_stats" in prompt


def test_system_prompt_contains_few_shot(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None)
    prompt = svc._build_system_prompt()
    assert "any dogs?" in prompt.lower() or "how many detections today" in prompt.lower()


# ---------------------------------------------------------------------------
# SUM(count) vs COUNT(*) — verified via raw SQL on test DB
# ---------------------------------------------------------------------------

def test_sum_count_vs_count_star(tmp_path):
    """SUM(count) should total objects; COUNT(*) should count rows."""
    alerts = [
        dict(ts=datetime(2026, 3, 29, 19, 0, tzinfo=timezone.utc).timestamp(),
             count=4, best_conf=0.88, image_path=None,
             trigger_classes=["person"], context_classes=["person"]),
        dict(ts=datetime(2026, 3, 29, 20, 0, tzinfo=timezone.utc).timestamp(),
             count=5, best_conf=0.75, image_path=None,
             trigger_classes=["person"], context_classes=["person"]),
    ]
    _, db_path = _make_db(tmp_path, alerts)
    conn = sqlite3.connect(db_path)
    sum_count = conn.execute("SELECT SUM(count) FROM alerts").fetchone()[0]
    count_star = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    conn.close()
    assert sum_count == 9   # 4 + 5
    assert count_star == 2  # 2 rows


# ---------------------------------------------------------------------------
# IST date boundary queries
# ---------------------------------------------------------------------------

def test_ist_date_boundary_includes_utc_previous_day(tmp_path):
    """2026-03-29T18:53 UTC = 2026-03-30 00:23 IST → should be in IST 'today' (Mar 30)."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE ts >= '2026-03-29T18:30:00' AND ts < '2026-03-30T18:30:00'"
    ).fetchone()
    conn.close()
    assert rows[0] == 6


def test_utc_midnight_excludes_ist_evening(tmp_path):
    """UTC midnight..midnight would miss IST evening entries — verifying the wrong range returns fewer."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE ts >= '2026-03-30T00:00:00' AND ts < '2026-03-31T00:00:00'"
    ).fetchone()
    conn.close()
    assert rows[0] == 2
