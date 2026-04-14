from unittest.mock import patch

from app.core.alert_policy import AlertPolicy


def test_flush_uses_latest_snapshot_metadata_when_available():
    policy = AlertPolicy(window_sec=1.0)

    policy.add(
        ids=[1],
        best_conf=0.60,
        now=0.0,
        rearm_sec=0.0,
        frame_img_path="/tmp/a.jpg",
        frame_count=1,
        frame_best_conf=0.60,
        frame_class_names={"person"},
        frame_context_class_names={"person", "car"},
    )
    policy.add(
        ids=[2],
        best_conf=0.90,
        now=0.2,
        rearm_sec=0.0,
        frame_img_path="/tmp/b.jpg",
        frame_count=2,
        frame_best_conf=0.90,
        frame_class_names={"person", "dog"},
        frame_context_class_names={"person", "dog", "car"},
    )

    count, best, img, classes, context_classes = policy.flush(now=1.5)
    assert count == 2
    assert best == 0.90
    assert img == "/tmp/b.jpg"
    assert classes == {"person", "dog"}
    assert context_classes == {"person", "dog", "car"}


def test_flush_falls_back_to_aggregated_count_without_snapshot():
    policy = AlertPolicy(window_sec=1.0)
    policy.add(ids=[1], best_conf=0.70, now=0.0, rearm_sec=0.0)
    policy.add(ids=[2], best_conf=0.80, now=0.1, rearm_sec=0.0)

    count, best, img, classes, context_classes = policy.flush(now=1.2)
    assert count == 2
    assert best == 0.80
    assert img is None
    assert classes == set()
    assert context_classes == set()


# ---------------------------------------------------------------------------
# Wall-clock cooldown
# ---------------------------------------------------------------------------

def _make_policy(window=5.0, cooldown=150.0, wall_time=1000.0):
    """Create a policy with a mocked wall-clock start time."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = wall_time
        p = AlertPolicy(window_sec=window, cooldown_sec=cooldown)
    return p


def test_wall_clock_seeded_on_init():
    """__post_init__ should seed _wall_last_sent to current wall-clock."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = 9999.0
        p = AlertPolicy(window_sec=5.0, cooldown_sec=60.0)
    assert p._wall_last_sent == 9999.0


def test_due_blocked_by_wall_clock_after_init():
    """Right after init, due() should be False even if pipeline clock is large."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = 1000.0
        p = AlertPolicy(window_sec=5.0, cooldown_sec=150.0)

        p.add(ids=[1], best_conf=0.9, now=200.0, rearm_sec=0.0)

        # Wall-clock says only 10s have passed (not enough for 150s cooldown)
        mock_time.time.return_value = 1010.0
        assert not p.due(now=200.0)


def test_due_passes_after_wall_cooldown():
    """due() should pass once both pipeline and wall-clock exceed cooldown."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = 1000.0
        p = AlertPolicy(window_sec=5.0, cooldown_sec=150.0)

        p.add(ids=[1], best_conf=0.9, now=200.0, rearm_sec=0.0)

        # Wall-clock has passed 150s
        mock_time.time.return_value = 1151.0
        assert p.due(now=200.0)


def test_flush_updates_wall_clock():
    """After flush(), _wall_last_sent should be updated to current wall time."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = 1000.0
        p = AlertPolicy(window_sec=5.0, cooldown_sec=60.0)

        p.add(ids=[1], best_conf=0.9, now=100.0, rearm_sec=0.0)
        mock_time.time.return_value = 1061.0
        assert p.due(now=100.0)

        mock_time.time.return_value = 2000.0
        p.flush(now=100.0)
        assert p._wall_last_sent == 2000.0


def test_new_track_id_cannot_bypass_global_cooldown():
    """A brand-new track ID should NOT fire an alert if global cooldown is active."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = 1000.0
        p = AlertPolicy(window_sec=5.0, cooldown_sec=150.0)

        # First alert
        p.add(ids=[1], best_conf=0.9, now=200.0, rearm_sec=20.0)
        mock_time.time.return_value = 1151.0
        assert p.due(now=200.0)
        p.flush(now=200.0)

        # 10s later, a NEW track id appears — should NOT bypass the 150s cooldown
        p.add(ids=[99], best_conf=0.9, now=210.0, rearm_sec=20.0)
        mock_time.time.return_value = 1161.0  # only 10s since flush
        assert not p.due(now=210.0), "New track ID should not bypass global cooldown"


def test_cooldown_zero_uses_window_sec_only():
    """With cooldown_sec=0, only window_sec should gate alerts."""
    with patch("app.core.alert_policy._time") as mock_time:
        mock_time.time.return_value = 1000.0
        p = AlertPolicy(window_sec=5.0, cooldown_sec=0.0)

        p.add(ids=[1], best_conf=0.9, now=100.0, rearm_sec=0.0)
        mock_time.time.return_value = 1006.0  # 6s wall > 5s window
        assert p.due(now=100.0)


def test_per_track_rearm_still_works():
    """The same track ID should not re-trigger within rearm_sec."""
    p = _make_policy(window=5.0, cooldown=0.0, wall_time=0.0)
    p._wall_last_sent = 0.0  # allow wall clock to pass

    p.add(ids=[1], best_conf=0.9, now=100.0, rearm_sec=20.0)
    p.flush(now=106.0)

    # Same track, only 5s later — should not be added
    p.add(ids=[1], best_conf=0.9, now=111.0, rearm_sec=20.0)
    assert not p.pending_ids, "Same track ID should be blocked by rearm"
