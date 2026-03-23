from datetime import datetime, timezone

from app.core.alert_history import AlertHistoryStore
from app.core.qa import QAService, _extract_sql, _execute_readonly_sql, _format_result_table


class FakeLLM:
    """LLM stub that returns canned SQL for known question patterns."""

    def __init__(self, sql: str = "", answer: str = ""):
        self._sql = sql
        self._answer = answer

    def complete(self, system_prompt: str, user_message: str) -> str:
        if "write a single sqlite select query" in system_prompt.lower():
            return self._sql
        if "fix the query" in system_prompt.lower():
            return self._sql
        return self._answer


def test_text_to_sql_end_to_end(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    store.insert_alert(
        ts=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc).timestamp(),
        count=2,
        best_conf=0.88,
        image_path=None,
        trigger_classes=["person"],
        context_classes=["person", "car"],
    )
    store.insert_alert(
        ts=datetime(2026, 3, 19, 14, 0, tzinfo=timezone.utc).timestamp(),
        count=1,
        best_conf=0.75,
        image_path=None,
        trigger_classes=["dog"],
        context_classes=["dog"],
    )

    llm = FakeLLM(
        sql="SELECT ts, count, trigger_classes FROM alerts ORDER BY ts",
        answer="There were 2 alerts: person at 15:30 IST and dog at 19:30 IST.",
    )
    svc = QAService(history=store, llm=llm)
    answer = svc.answer_question("what were all the detections?")

    assert "2 alerts" in answer
    assert "person" in answer
    assert "dog" in answer


def test_no_llm_returns_config_message(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    svc = QAService(history=store, llm=None)
    answer = svc.answer_question("how many alerts today?")
    assert "LLM is not configured" in answer


def test_extract_sql_strips_markdown_fences():
    raw = "```sql\nSELECT * FROM alerts LIMIT 10\n```"
    assert _extract_sql(raw) == "SELECT * FROM alerts LIMIT 10"


def test_extract_sql_rejects_destructive_statements():
    assert _extract_sql("DROP TABLE alerts") is None
    assert _extract_sql("DELETE FROM alerts") is None
    assert _extract_sql("INSERT INTO alerts VALUES(1)") is None
    assert _extract_sql("SELECT 1; DROP TABLE alerts") is None


def test_extract_sql_allows_valid_select():
    sql = "SELECT ts, count FROM alerts WHERE ts >= '2026-03-19' LIMIT 50"
    assert _extract_sql(sql) == sql


def test_execute_readonly_sql_reads_data(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    store.insert_alert(
        ts=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc).timestamp(),
        count=3,
        best_conf=0.9,
        image_path=None,
        trigger_classes=["person"],
    )

    rows, cols, err = _execute_readonly_sql(str(tmp_path / "alerts.db"), "SELECT count FROM alerts")
    assert err is None
    assert len(rows) == 1
    assert rows[0][0] == 3


def test_execute_readonly_sql_blocks_writes(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    _, _, err = _execute_readonly_sql(
        str(tmp_path / "alerts.db"),
        "INSERT INTO alerts(ts, count, best_conf) VALUES('x', 1, 0.5)",
    )
    assert err is not None


def test_format_result_table_empty():
    assert _format_result_table([], []) == "(no results)"


def test_format_result_table_with_rows():
    cols = ["ts", "count"]
    rows = [("2026-03-19T10:00:00", 3)]
    text = _format_result_table(cols, rows)
    assert "ts" in text
    assert "2026-03-19T10:00:00" in text
    assert "3" in text


def test_sql_retry_on_error(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    store.insert_alert(
        ts=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc).timestamp(),
        count=2,
        best_conf=0.8,
        image_path=None,
    )

    call_count = [0]

    class RetryLLM:
        def complete(self, system_prompt: str, user_message: str) -> str:
            call_count[0] += 1
            if "write a single sqlite select query" in system_prompt.lower():
                return "SELECT count FROM alertsss"
            if "fix the query" in system_prompt.lower():
                return "SELECT count FROM alerts"
            return "There were 2 detected objects."

    svc = QAService(history=store, llm=RetryLLM())
    answer = svc.answer_question("how many objects?")
    assert "2" in answer
