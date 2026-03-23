from datetime import datetime, timezone

from app.core.alert_history import AlertHistoryStore
from app.core.qa import QAService


def test_last_alert_is_reported_in_ist(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    store.insert_alert(ts=0.0, count=1, best_conf=0.90, image_path=None)

    answer = QAService(history=store).answer_question("When was the last alert?")

    assert "1970-01-01 05:30:00 IST" in answer
    assert "IST" in answer
    assert "UTC" not in answer


def test_iso_date_query_uses_ist_day_window(tmp_path):
    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    # Included in 2026-03-19 IST (2026-03-18 20:00 UTC == 2026-03-19 01:30 IST)
    store.insert_alert(
        ts=datetime(2026, 3, 18, 20, 0, tzinfo=timezone.utc).timestamp(),
        count=2,
        best_conf=0.8,
        image_path=None,
    )
    # Excluded from 2026-03-19 IST (2026-03-19 20:00 UTC == 2026-03-20 01:30 IST)
    store.insert_alert(
        ts=datetime(2026, 3, 19, 20, 0, tzinfo=timezone.utc).timestamp(),
        count=4,
        best_conf=0.9,
        image_path=None,
    )

    answer = QAService(history=store).answer_question("How many people came on 2026-03-19?")

    assert "On 2026-03-19 (IST), there were 1 alerts with a total of 2 detected objects." in answer


def test_llm_infers_date_when_natural_phrase_has_no_explicit_date(tmp_path):
    class FakeLLM:
        def complete(self, system_prompt: str, user_message: str) -> str:
            if "Return only JSON with keys: intent, date." in system_prompt:
                return '{"intent":"date_query","date":"2026-03-19"}'
            return "On 2026-03-19 (IST), there were 1 alerts with a total of 3 detected objects."

    store = AlertHistoryStore(str(tmp_path / "alerts.db"))
    store.insert_alert(
        ts=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc).timestamp(),
        count=3,
        best_conf=0.95,
        image_path=None,
    )

    answer = QAService(history=store, llm=FakeLLM()).answer_question("how many visitors were there that day?")

    assert "On 2026-03-19 (IST), there were 1 alerts with a total of 3 detected objects." in answer
