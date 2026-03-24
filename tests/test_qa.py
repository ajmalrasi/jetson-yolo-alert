from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.core.alert_history import AlertHistoryStore
from app.core.qa import QAService


def _make_db(tmp_path, alerts=None):
    db_path = str(tmp_path / "alerts.db")
    store = AlertHistoryStore(db_path)
    for alert in alerts or []:
        store.insert_alert(**alert)
    return store, db_path


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
