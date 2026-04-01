# QA and Telegram question logging

Human-readable summary for git. Cursor loads the same content from `.cursor/rules/qa-telegram-debug-logging.mdc` when that file exists (create it from the block at the bottom if missing).

## Where it is saved

Natural-language Q&A (including Telegram) is logged by the QA service to:

| Item | Value |
|------|--------|
| **Logger name** | `qa.trace` |
| **File** | `{SAVE_DIR}/qa_trace.log` |
| **Default `SAVE_DIR`** | `/workspace/work/alerts` in Docker; host often `./work/alerts` via compose |

Configure `SAVE_DIR` in `.env`. Handler setup: `app/core/qa.py` at import time.

## What is logged (per question)

Each successful `QAService.answer_question(...)` writes one line with:

- **Q:** user question
- **UTC / IST** request timestamps
- **SQL** generated query
- **Result** formatted SQL output
- **Answer** text returned to the user

Errors use `logger.exception` in `answer_question`.

## Telegram

There is **no separate Telegram message store**. The bot (`app/adapters/chat_telegram_bot.py`) calls `qa_service.answer_question(question)`; the transcript for debugging is **`qa_trace.log`**.

## For agents debugging QA

1. Read `{SAVE_DIR}/qa_trace.log`.
2. Resolve `SAVE_DIR` from the runtime environment if needed.
3. Detection data is in SQLite (`alert_history.db` / `cfg.alert_db_path`), not the Q&A log.

## Cursor rule file (copy to `.cursor/rules/qa-telegram-debug-logging.mdc`)

Create this file so future agents always see the reference (`alwaysApply: true`):

```markdown
---
description: Where Telegram Q&A questions and QA debug traces are persisted for debugging
alwaysApply: true
---

# QA and Telegram question logging

## Where it is saved

All natural-language Q&A (including messages from Telegram) is logged by the QA service to a **single append-only log file**:

| Item | Value |
|------|--------|
| **Logger name** | `qa.trace` (Python `logging` logger) |
| **File path** | `{SAVE_DIR}/qa_trace.log` |
| **Default `SAVE_DIR`** | `/workspace/work/alerts` inside Docker; on the host this is usually `./work/alerts` when using the default `docker-compose` volume mount |

Set `SAVE_DIR` in `.env` to change the directory. The handler is created at import time in `app/core/qa.py` (`FileHandler` on `qa_trace.log`).

## What is logged (per question)

Each successful `QAService.answer_question(...)` call writes one **DEBUG** line containing:

- **Q:** the user’s question text (same string Telegram passed in)
- **UTC:** and **IST:** timestamps at request time
- **SQL:** the generated SQL query
- **Result:** formatted SQL result text passed to the answer step
- **Answer:** the final natural-language reply sent to the user

Failures are also logged via the standard module logger (`logger.exception` on errors in `answer_question`).

## Telegram-specific storage

There is **no separate Telegram chat history table or JSON file** in this repo for “all Telegram messages.” The bot calls `qa_service.answer_question(question)` in a thread; persistence for debugging is **`qa_trace.log` only** (plus Docker/container logs if you capture stdout).

## How another agent should use this

1. Read **`{SAVE_DIR}/qa_trace.log`** (tail or full file) to correlate user questions with SQL, raw results, and answers.
2. Confirm **`SAVE_DIR`** from the running environment (`.env` / `docker compose exec` env) if paths differ from defaults.
3. Alert history lives in SQLite at **`cfg.alert_db_path`** (default under `SAVE_DIR`, e.g. `alert_history.db`) — that is detections data, not the Q&A transcript.

## Code reference

Logging is implemented in `app/core/qa.py`: `qa_trace.debug(...)` after SQL execution and answer formatting.
```
