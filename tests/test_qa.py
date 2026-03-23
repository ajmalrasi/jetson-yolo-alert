from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from langchain_community.utilities import SQLDatabase

from app.core.alert_history import AlertHistoryStore
from app.core.qa import QAService


def _make_db(tmp_path, alerts=None):
    """Create an AlertHistoryStore with optional seed data and return both
    the store and a SQLDatabase instance pointing to the same file."""
    db_path = str(tmp_path / "alerts.db")
    store = AlertHistoryStore(db_path)
    for alert in alerts or []:
        store.insert_alert(**alert)
    sql_db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    return store, sql_db


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


def test_answer_question_calls_agent(tmp_path):
    """QAService should delegate to create_sql_agent and return its output."""
    _, sql_db = _make_db(tmp_path, _seed_two_alerts())
    fake_llm = MagicMock()

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "output": "There were 2 alerts: person at 15:30 IST and dog at 19:30 IST."
    }

    svc = QAService(db=sql_db, llm=fake_llm)
    with patch("app.core.qa.create_sql_agent", return_value=fake_agent) as mock_create:
        answer = svc.answer_question("what were all the detections?")

    mock_create.assert_called_once()
    fake_agent.invoke.assert_called_once()
    assert "2 alerts" in answer
    assert "person" in answer


def test_no_llm_returns_config_message(tmp_path):
    _, sql_db = _make_db(tmp_path)
    svc = QAService(db=sql_db, llm=None)
    answer = svc.answer_question("how many alerts today?")
    assert "LLM is not configured" in answer


def test_empty_question_returns_message(tmp_path):
    _, sql_db = _make_db(tmp_path)
    fake_llm = MagicMock()
    svc = QAService(db=sql_db, llm=fake_llm)
    answer = svc.answer_question("   ")
    assert "Please provide a question" in answer


def test_agent_exception_returns_error(tmp_path):
    """If the SQL agent raises, the user gets a friendly error instead of a traceback."""
    _, sql_db = _make_db(tmp_path, _seed_two_alerts())
    fake_llm = MagicMock()

    def _explode(*_a, **_kw):
        raise RuntimeError("LLM connection failed")

    svc = QAService(db=sql_db, llm=fake_llm)
    with patch("app.core.qa.create_sql_agent", side_effect=_explode):
        answer = svc.answer_question("how many alerts?")

    assert "went wrong" in answer.lower()


def test_db_has_expected_tables(tmp_path):
    """The SQLDatabase wrapper should see the 'alerts' table."""
    _, sql_db = _make_db(tmp_path, _seed_two_alerts())
    assert "alerts" in sql_db.get_usable_table_names()
