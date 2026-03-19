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
    )
    policy.add(
        ids=[2],
        best_conf=0.90,
        now=0.2,
        rearm_sec=0.0,
        frame_img_path="/tmp/b.jpg",
        frame_count=2,
        frame_best_conf=0.90,
    )

    count, best, img = policy.flush(now=1.5)
    assert count == 2
    assert best == 0.90
    assert img == "/tmp/b.jpg"


def test_flush_falls_back_to_aggregated_count_without_snapshot():
    policy = AlertPolicy(window_sec=1.0)
    policy.add(ids=[1], best_conf=0.70, now=0.0, rearm_sec=0.0)
    policy.add(ids=[2], best_conf=0.80, now=0.1, rearm_sec=0.0)

    count, best, img = policy.flush(now=1.2)
    assert count == 2
    assert best == 0.80
    assert img is None
