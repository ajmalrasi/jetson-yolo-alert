from app.core.pipeline import _build_alert_message


def test_alert_message_plural_and_classes():
    msg = _build_alert_message(count=2, best=0.91, trigger_class_names={"person", "dog"})
    assert "Alert detected" in msg
    assert "Triggered: 2 objects" in msg
    assert "Best confidence: 0.91" in msg
    assert "Trigger classes: dog, person" in msg


def test_alert_message_singular():
    msg = _build_alert_message(count=1, best=0.72, trigger_class_names={"person"})
    assert "Triggered: 1 object" in msg
    assert "Best confidence: 0.72" in msg
    assert "Trigger classes: person" in msg
