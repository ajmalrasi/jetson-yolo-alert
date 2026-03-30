from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.core.alert_history import AlertHistoryStore
from app.core.qa import QAService, _utc_results_to_ist


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
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT ts FROM alerts").fetchall()
    conn.close()
    for (ts,) in rows:
        assert "+00:00" not in ts, f"Timestamp still has offset: {ts}"
        assert "T" in ts, f"Unexpected format: {ts}"


def test_legacy_offset_migrated_on_init(tmp_path):
    """Opening the DB should strip +00:00 from any legacy rows."""
    db_path = str(tmp_path / "alerts.db")
    import sqlite3
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
# UTC → IST conversion for answer LLM
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
# SQL execution: row count prefix & image extraction
# ---------------------------------------------------------------------------

def test_execute_sql_detection_count_prefix(tmp_path):
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql("SELECT id, ts FROM alerts;")
    assert result.startswith("(6 detections found)")


def test_execute_sql_no_rows(tmp_path):
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, img = svc._execute_sql("SELECT * FROM alerts WHERE count > 999;")
    assert "0 matches" in result
    assert img is None


def test_execute_sql_extracts_first_image(tmp_path):
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    _, img = svc._execute_sql(
        "SELECT image_path FROM alerts WHERE trigger_classes LIKE '%dog%' ORDER BY ts;"
    )
    assert img == "/workspace/work/alerts/snap_dog1.jpg"


# ---------------------------------------------------------------------------
# Class filtering queries against real data
# ---------------------------------------------------------------------------

def test_dog_query_returns_2(tmp_path):
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql(
        "SELECT * FROM alerts WHERE trigger_classes LIKE '%\"dog\"%' OR context_classes LIKE '%\"dog\"%';"
    )
    assert "(2 detections found)" in result


def test_person_query_returns_3(tmp_path):
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql(
        "SELECT * FROM alerts WHERE trigger_classes LIKE '%\"person\"%' OR context_classes LIKE '%\"person\"%';"
    )
    assert "(3 detections found)" in result


def test_motorcycle_in_context_only(tmp_path):
    """Motorcycle appears only in context_classes, not trigger_classes."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql(
        "SELECT * FROM alerts WHERE trigger_classes LIKE '%\"motorcycle\"%';"
    )
    assert "0 matches" in result
    result2, _ = svc._execute_sql(
        "SELECT * FROM alerts WHERE context_classes LIKE '%\"motorcycle\"%';"
    )
    assert "(1 detections found)" in result2


# ---------------------------------------------------------------------------
# IST date boundary queries
# ---------------------------------------------------------------------------

def test_ist_date_boundary_includes_utc_previous_day(tmp_path):
    """2026-03-29T18:53 UTC = 2026-03-30 00:23 IST → should be in IST 'today' (Mar 30)."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql(
        "SELECT * FROM alerts WHERE ts >= '2026-03-29T18:30:00' AND ts < '2026-03-30T18:30:00';"
    )
    assert "(6 detections found)" in result


def test_utc_midnight_excludes_ist_evening(tmp_path):
    """UTC midnight..midnight would miss IST evening entries — verifying the wrong range returns fewer."""
    _, db_path = _make_db(tmp_path, _make_real_day_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql(
        "SELECT * FROM alerts WHERE ts >= '2026-03-30T00:00:00' AND ts < '2026-03-31T00:00:00';"
    )
    assert "(2 detections found)" in result


# ---------------------------------------------------------------------------
# Original tests (kept for backwards compat)
# ---------------------------------------------------------------------------

def _seed_two_alerts():
    return [
        dict(
            ts=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc).timestamp(),
            count=2,
            best_conf=0.88,
            image_path=None,
            trigger_classes=["person"],
            context_classes=["person", "car"],
        ),
        dict(
            ts=datetime(2026, 3, 19, 14, 0, tzinfo=timezone.utc).timestamp(),
            count=1,
            best_conf=0.75,
            image_path=None,
            trigger_classes=["dog"],
            context_classes=["dog"],
        ),
    ]


def test_generate_sql_and_answer(tmp_path):
    """QAService calls LLM twice: once for SQL, once for answer formatting."""
    _, db_path = _make_db(tmp_path, _seed_two_alerts())

    fake_llm = MagicMock()
    sql_response = MagicMock()
    sql_response.content = "SELECT COUNT(*) as total FROM alerts;"
    answer_response = MagicMock()
    answer_response.content = "There were 2 alerts today."
    fake_llm.invoke.side_effect = [sql_response, answer_response]

    svc = QAService(db_path=db_path, llm=fake_llm)
    answer = svc.answer_question("how many alerts?")

    assert fake_llm.invoke.call_count == 2
    assert "2 alerts" in answer.text


def test_no_llm_returns_config_message(tmp_path):
    _, db_path = _make_db(tmp_path)
    svc = QAService(db_path=db_path, llm=None)
    answer = svc.answer_question("how many alerts today?")
    assert "LLM is not configured" in answer.text


def test_empty_question_returns_message(tmp_path):
    _, db_path = _make_db(tmp_path)
    fake_llm = MagicMock()
    svc = QAService(db_path=db_path, llm=fake_llm)
    answer = svc.answer_question("   ")
    assert "Please provide a question" in answer.text


def test_unsafe_sql_blocked(tmp_path):
    """If LLM generates a destructive query, it should be blocked."""
    _, db_path = _make_db(tmp_path, _seed_two_alerts())

    fake_llm = MagicMock()
    sql_response = MagicMock()
    sql_response.content = "DROP TABLE alerts;"
    fake_llm.invoke.return_value = sql_response

    svc = QAService(db_path=db_path, llm=fake_llm)
    answer = svc.answer_question("delete everything")
    assert "went wrong" in answer.text.lower()


def test_execute_sql_returns_rows(tmp_path):
    """_execute_sql should return formatted table rows."""
    _, db_path = _make_db(tmp_path, _seed_two_alerts())
    svc = QAService(db_path=db_path, llm=None)
    result, _ = svc._execute_sql("SELECT count, trigger_classes FROM alerts ORDER BY ts;")
    assert "count" in result
    assert "person" in result
    assert "dog" in result
