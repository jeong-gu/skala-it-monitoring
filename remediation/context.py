"""Persist the last structured investigation without altering the message schema."""
import json


def remember(store, thread_id, result, query):
    if not result.get("evidence") or not hasattr(store, "conn"): return
    store.conn.execute("CREATE TABLE IF NOT EXISTS investigation_context (thread_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    data = {k:result.get(k, []) for k in ("evidence", "tasks", "hypotheses", "limitations")}
    data["query"] = query
    data["subject_query"] = result.get("request_subject") or query
    data["manual_requirements"]=[{k:r[k] for k in ("description","information_ids")} for r in result.get("requirements",[]) if r["kind"]=="remediation"]
    data["scenario_id"] = result.get("scenario_id")
    data["had_remediation"] = (result.get("remediation") or {}).get("status")=="provided"
    store.conn.execute("INSERT INTO investigation_context VALUES (?,?) ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload",
                       (thread_id,json.dumps(data,ensure_ascii=False)))
    store.conn.commit()


def recall(store, thread_id):
    if not hasattr(store, "conn"): return None
    found=store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='investigation_context'").fetchone()
    if not found: return None
    row=store.conn.execute("SELECT payload FROM investigation_context WHERE thread_id=?",(thread_id,)).fetchone()
    return json.loads(row[0]) if row else None
