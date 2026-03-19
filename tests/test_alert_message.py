from app.core.pipeline import _build_alert_message


def test_alert_message_plural_and_classes():
    msg = _build_alert_message(
        count=2,
        best=0.91,
        trigger_class_names={"person", "dog"},
        context_class_names={"person", "dog", "car"},
    )
    assert "Alert detected" in msg
    assert "Triggered: 2 objects" in msg
    assert "Best confidence: 0.91" in msg
    assert "Triggered classes: dog, person" in msg
    assert "Context classes in image: car, dog, person" in msg


def test_alert_message_singular():
    msg = _build_alert_message(
        count=1,
        best=0.72,
        trigger_class_names={"person"},
        context_class_names=set(),
    )
    assert "Triggered: 1 object" in msg
    assert "Best confidence: 0.72" in msg
    assert "Triggered classes: person" in msg
    assert "Context classes in image" not in msg
