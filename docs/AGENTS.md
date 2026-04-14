# Agent instructions — QA & Telegram `/ask` debugging

**Single reference** for assistants working on this repo. Product overview for people: `README.md`.

---

## Where it is saved

Natural-language Q&A (CLI and Telegram) is logged by the QA service:

| Item | Value |
|------|--------|
| **Logger** | `qa.trace` (`logging.getLogger("qa.trace")`) |
| **File** | `{SAVE_DIR}/qa_trace.log` |
| **Default `SAVE_DIR`** | `/workspace/work/alerts` in Docker; with `./work:/workspace/work` the host path is usually `./work/alerts` |

Set `SAVE_DIR` in `.env`. Handler is created at import time in `app/core/qa.py` (`FileHandler` on `qa_trace.log`).

## What is logged (per successful question)

Each `QAService.answer_question(...)` emits a **DEBUG** record via `qa_trace.debug(...)` containing:

- **Q:** exact user question (Telegram passes the same string)
- **UTC / IST** timestamps used in the SQL prompt
- **SQL** generated query
- **Result:** formatted table text passed to the answer LLM (may contain embedded newlines → many physical lines in the file)
- **Answer:** text returned to the user

Errors: `logger.exception` in `answer_question`.

## Telegram

**No separate Telegram message store or chat-history table.** `app/adapters/chat_telegram_bot.py` calls `qa_service.answer_question(question)`; the only persisted transcript for Q&A is **`qa_trace.log`** (plus container stdout if captured).

## How to investigate

1. Resolve **`SAVE_DIR`** (`.env`, or `docker compose exec ask-telegram printenv SAVE_DIR`).
2. Read **`{SAVE_DIR}/qa_trace.log`** — `tail -n 300`, or `grep " Q: "` / `rg " Q: "` for question lines.
3. **Alert rows** live in SQLite at **`ALERT_DB_PATH`** (default under `SAVE_DIR`, e.g. `alert_history.db`). That is detection data; **`qa_trace.log` is not a substitute** for SQL ground truth.

## Interpreting “wrong” or contradictory answers

1. **Compare `ts` bounds in SQL** (`ts >= '...' AND ts < '...'`) across questions. Scopes like “today”, “since 10am”, “this week” yield different windows — counts may all be valid without matching.
2. **`(N detections found)`** — `N` is **rows returned** by SQLite (capped by `MAX_RESULT_ROWS` in `qa.py`). For **`SELECT COUNT(*)`** you get **one result row**; the prefix may say `(1 detections found)` while the **`COUNT(*)`** column is the real total. Read the aggregate value, not only the prefix.
3. **`class_names` / LIKE filters** — dog vs person counts can differ when one question is time-bounded and another is “whole day”.

## `/describe` trace log

The VLM video understanding feature has its own trace log:

| Item | Value |
|------|--------|
| **Logger** | `describe.trace` (`logging.getLogger("describe.trace")`) |
| **File** | `{SAVE_DIR}/describe_trace.log` |

Each `/describe` query logs: the user query, parsed time range (UTC), number of frames queried/sampled, VLM model used, and the full narrative response. Created at import time in `app/core/video_understanding.py`.

## Code entrypoints

| Concern | File |
|--------|------|
| QA: SQL, execute, answer, trace | `app/core/qa.py` |
| DB + LLM wiring | `app/core/qa_factory.py` |
| VLM: time parsing, sampling, describe trace | `app/core/video_understanding.py` |
| Telegram `/ask` + `/describe` | `app/adapters/chat_telegram_bot.py` |

## Cursor

Project rule **`.cursor/rules/qa-telegram-debug-logging.mdc`** (`alwaysApply: true`) summarizes persistence and points here for full workflow — keep that file’s “see also” aligned with this doc if you edit either.
