# Action items – functional improvements

Use this file to track what you want to do. Mark items when you start or finish them.

**How to manage:**
- `[ ]` = not started  
- `[~]` = in progress  
- `[x]` = done  
- Add a `(priority: high/medium/low)` or due date if you like.

---

## Detection & alerts

- [ ] **Zone / ROI** – Only trigger when detection is inside a defined polygon (e.g. driveway, door).
- [ ] **“Left the scene” alert** – Optional notification when a previously seen object is no longer detected.
- [ ] **Class-specific thresholds** – Different confidence or persistence per class (e.g. stricter for person).
- [ ] **Quiet hours / schedule** – No alerts (or different sensitivity) during certain times (e.g. night, when home).

---

## Notifications & UX

- [ ] **Rich notifications** – Send a short clip or 2–3 frames instead of a single snapshot.
- [ ] **Severity or summary** – Different message/channel for “first seen” vs “still present” vs “left”.
- [ ] **Ack / mute** – Way to acknowledge an alert or mute for N minutes.

---

## Reliability & operations

- [ ] **Camera / stream health** – Detect “no frames” or “stream died” and send a “camera offline” alert.
- [ ] **Reconnection** – After RTSP/network drop, retry opening the stream with backoff.
- [ ] **Graceful shutdown** – On stop, finish pending alert and release camera cleanly.

---

## Configuration & tuning

- [ ] **Per-camera profiles** – Different config (trigger classes, zones, sensitivity) per camera.
- [ ] **Sensitivity preset** – “Low / normal / high” that maps to confidence + persistence + rate.

---

## Security & privacy

- [ ] **Snapshot retention** – Auto-delete or overwrite old snapshots after N days or N files.
- [ ] **Optional blur/redact** – Blur faces or regions in saved images before save/send.

---

## Observability

- [ ] **Simple dashboard or status page** – “Last frame time”, “last alert”, “current FPS”, “errors in last hour”.
- [ ] **Alert history log** – Append-only log (file or SQLite) of when alerts fired and why.

---

*Last updated: 2026-02-26*
