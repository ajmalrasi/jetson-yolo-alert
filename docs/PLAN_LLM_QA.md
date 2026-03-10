# Plan: LLM-powered Q&A over alert history

Goal: answer natural-language questions like *"How many people came on March 10?"* or *"When was the last alert?"* using an LLM and stored alert data.

---

## 1. What we need

| Need | Purpose |
|------|--------|
| **Stored alert history** | So we can query "alerts on date X" or "count of alerts per day". Right now alerts are only sent to Telegram and not persisted. |
| **LLM** | To understand the question and produce an answer (optionally with a step that turns the question into a query). |
| **Interface** | Way for you to ask (e.g. Telegram bot, small API, or CLI). |

---

## 2. Data layer – store alert history

- **Where:** SQLite (single file, no extra server, works well on Jetson).
- **Location:** e.g. `SAVE_DIR/alert_history.db` or `workspace/work/alert_history.db`.
- **Schema (minimal):**
  - **alerts**  
    `id`, `ts` (UTC datetime), `count` (number of objects in that alert), `best_conf`, `image_path` (optional).  
  - Optional later: **presence_events** (arrive/leave) for “how many distinct visits” if you want that.
- **When to write:** On every alert (in the pipeline right after sending to Telegram, or via an event-bus subscriber that writes to DB). Same process as the alert service.
- **Existing action item:** *"Alert history log – Append-only log (file or SQLite)"* in `ACTION_ITEMS.md` covers this; implementing it is step 1 for LLM Q&A.

**Tasks:**
- [ ] Add SQLite DB creation and `alerts` table (migration or init script).
- [ ] Wire pipeline (or event subscriber) to insert a row per alert (`ts`, `count`, `best_conf`, `image_path`).
- [ ] Add a small **query module** (e.g. `app/core/alert_history.py`): functions like `get_alerts_between(start_ts, end_ts)`, `get_alerts_on_date(date_str)` returning list of records. No LLM yet.

---

## 3. LLM – how it answers

Two possible patterns:

**A) Query-then-answer (recommended)**  
1. User asks: *"How many people came on March 10?"*  
2. **We** interpret the question (or use the LLM once) to get a date → e.g. March 10, 2025.  
3. We **query the DB**: e.g. `get_alerts_on_date("2025-03-10")` → list of alerts (each has `count`, `ts`).  
4. We compute or aggregate: e.g. total objects = sum of `count` per alert, or “number of alerts” = len(list).  
5. We send **question + data** to the LLM: *"User asked: ... Given this data: [table or summary]. Give a short, direct answer."*  
6. LLM returns one sentence; we show that to the user.

**B) LLM does everything**  
1. User asks in natural language.  
2. We give the LLM the **raw data** (e.g. all alerts in a date range or summary stats).  
3. LLM both “understands” the question and writes the answer from the data.  
4. More flexible for odd phrasings; a bit more data/tokens and dependency on LLM reasoning.

**Recommendation:** Start with **A**: simple date parsing (or one LLM call to extract “which date?”), then deterministic query + aggregation, then one LLM call with “question + summary/table” to produce a friendly answer. That keeps logic clear and reduces cost/latency.

**LLM options on Jetson:**
- **Cloud API** (OpenAI, Groq, etc.): easy, good quality; needs API key and network.
- **Local** (Ollama, llama.cpp, TensorRT-LLM): no key, private; needs enough RAM/GPU and model choice.

**Tasks:**
- [ ] Choose LLM (e.g. OpenAI API for first version, or Ollama if you prefer local).
- [ ] Add a thin **LLM client** (e.g. `app/adapters/llm_openai.py` or `llm_ollama.py`): one function `complete(system_prompt, user_message)` → string.
- [ ] Add **QA service** (e.g. `app/core/qa.py`): `answer_question(question: str) -> str` that (1) optionally uses LLM to extract date/intent, (2) queries DB, (3) builds a small summary, (4) calls LLM with question + data, (5) returns answer.

---

## 4. Interface – how you ask

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **Telegram bot** | You already use Telegram for alerts; ask in the same chat. | Need to handle bot commands and maybe long-running process. |
| **CLI** | Simple: `python -m app.tools.ask "How many people on March 10?"` | You have to SSH or run on the device. |
| **Small REST API** | One endpoint `POST /ask` with `{"question": "..."}`; can be called from Telegram bot, web, or curl. | Need to run a small server (e.g. FastAPI) and secure it. |

**Recommendation:** Start with **CLI** (e.g. `python -m app.tools.ask "..."`) so the pipeline stays “alert-only” and you can run Q&A on demand. Add a **Telegram bot** that forwards messages to the same QA service later, so you can ask from your phone.

**Tasks:**
- [ ] CLI: `app/tools/ask.py` – reads question from argv or stdin, calls QA service, prints answer.
- [ ] (Later) Telegram bot: listen for “/ask …” or “ask …”, call QA service, reply in chat.
- [ ] (Optional) FastAPI app with `POST /ask` if you want a small API.

---

## 5. Implementation order

1. **DB + alert persistence**  
   - SQLite + table, write on each alert.  
   - Query helpers: by date, by time range.

2. **LLM client**  
   - One adapter (e.g. OpenAI or Ollama), `complete(system, user)`.

3. **QA service**  
   - `answer_question(question)` using: date extraction (simple or LLM), DB query, aggregation, then LLM to turn data + question into answer.

4. **CLI**  
   - `ask.py` that calls the QA service and prints the result.

5. **(Later)** Telegram bot or API so you can ask from Telegram or another client.

---

## 6. Example flow (end-to-end)

1. Pipeline runs; when an alert fires it writes a row to `alerts` (e.g. `ts=2025-03-10 14:32:00`, `count=1`, `best_conf=0.92`).
2. You run: `python -m app.tools.ask "How many people came on March 10?"`
3. QA service parses “March 10” → date 2025-03-10 (or asks LLM to resolve “yesterday” etc.).
4. Query: `get_alerts_on_date("2025-03-10")` → e.g. 12 rows, total `count` = 15.
5. Prompt to LLM: “User asked: How many people came on March 10? Data: 12 alerts on that day, total 15 object detections. Give a one-sentence answer.”
6. LLM: “On March 10 there were 12 alerts with a total of 15 person detections.”
7. CLI prints that and exits.

---

## 7. Config / env (to add later)

- `ALERT_DB_PATH` – path to SQLite file (default next to `SAVE_DIR`).
- `OPENAI_API_KEY` or `OLLAMA_HOST` – depending on which LLM adapter you use.
- Optional: `LLM_MODEL` (e.g. `gpt-4o-mini` or `llama3`).

---

*This plan assumes “how many people on date X” means: either (a) total number of object counts in alerts on that day, or (b) number of alerts on that day. The exact meaning can be clarified in the QA prompt (e.g. “count of alerts” vs “sum of detected objects”).*
