import time
from typing import Optional, List, Dict, Any
import httpx
from mcp.server.fastmcp import FastMCP

from mim.config import CONFIG

mcp = FastMCP("mim")

API_BASE = f"http://{CONFIG.api_host}:{CONFIG.api_port}"

_fallback_sql = None
_fallback_lance = None
_fallback_rag = None
_fallback_vss = None
_fallback_sum = None


def _get_local_fallbacks():
    global _fallback_sql, _fallback_lance, _fallback_rag, _fallback_vss, _fallback_sum
    if _fallback_sql is None:
        from mim.storage.sqlite_manager import SQLiteManager
        from mim.storage.lancedb_manager import LanceDBManager
        from mim.search.hybrid_rag import HybridRAGEngine
        from mim.system.vss_manager import VSSManager
        from mim.ai.gemini_summarizer import GeminiSummarizer

        _fallback_sql = SQLiteManager()
        _fallback_lance = LanceDBManager()
        _fallback_rag = HybridRAGEngine(sqlite_mgr=_fallback_sql, lancedb_mgr=_fallback_lance)
        _fallback_vss = VSSManager()
        _fallback_sum = GeminiSummarizer(sqlite_mgr=_fallback_sql)
    return _fallback_sql, _fallback_rag, _fallback_vss, _fallback_sum


@mcp.tool()
def search_screen_memory(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search the developer's historical desktop screen memory (browser error toasts, terminal logs,
    documentation, IDE viewports) using Hybrid TD-RRF.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{API_BASE}/search", params={"q": query, "limit": limit})
            if res.status_code == 200:
                return res.json().get("results", [])
    except Exception:
        pass

    _, rag, _, _ = _get_local_fallbacks()
    return rag.search(query=query, limit=limit)


@mcp.tool()
def get_recent_screen_activity(minutes: int = 15, limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve the developer's recent active windows, apps, and timestamps from the last N minutes."""
    now_ms = int(time.time() * 1000)
    start_ts = now_ms - (minutes * 60 * 1000)
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                f"{API_BASE}/timeline",
                params={"start_ts": start_ts, "end_ts": now_ms, "limit": limit},
            )
            if res.status_code == 200:
                return res.json().get("frames", [])
    except Exception:
        pass

    sql, _, _, _ = _get_local_fallbacks()
    conn = sql._get_conn()
    rows = conn.execute(
        "SELECT frame_id, timestamp_ms, process_name, window_title, vss_snapshot_id "
        "FROM frames WHERE timestamp_ms >= ? ORDER BY timestamp_ms DESC LIMIT ?",
        (start_ts, limit),
    ).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def get_latest_errors_from_screen(minutes: int = 15) -> List[Dict[str, Any]]:
    """
    Automatically scans the developer's recent Chrome browser tabs, terminals, and IDE windows
    for stack traces, exceptions, failed builds, or error messages.
    """
    now_ms = int(time.time() * 1000)
    start_ts = now_ms - (minutes * 60 * 1000)
    error_query = "error exception traceback fatal failed undefined status 500 404 crash syntaxerror"
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(
                f"{API_BASE}/search",
                params={"q": error_query, "limit": 12, "start_ts": start_ts, "end_ts": now_ms},
            )
            if res.status_code == 200:
                return res.json().get("results", [])
    except Exception:
        pass

    _, rag, _, _ = _get_local_fallbacks()
    return rag.search(query=error_query, limit=12, start_ts=start_ts, end_ts=now_ms)


@mcp.tool()
def prepare_safe_agent_edit(folder_path: str) -> Dict[str, Any]:
    """
    Call this BEFORE performing multi-file refactors or risky shell commands.
    Creates a rolling backup snapshot of `folder_path` and returns `snapshot_id`.
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.post(
                f"{API_BASE}/vss/snapshot",
                json={"target_path": folder_path, "is_auto": True, "skip_if_unchanged": False},
            )
            if res.status_code == 200:
                return res.json()
    except Exception:
        pass

    _, _, vss, _ = _get_local_fallbacks()
    return vss.create_snapshot(target_path=folder_path, is_auto=True)


@mcp.tool()
def create_workspace_backup(folder_path: str) -> Dict[str, Any]:
    """Alias for creating an explicit manual workspace backup snapshot."""
    return prepare_safe_agent_edit(folder_path=folder_path)


@mcp.tool()
def undo_last_agent_edit(
    snapshot_id: Optional[str] = None,
    restore_target_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Restores the workspace back to `snapshot_id` (or the most recent backup if omitted)."""
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.post(
                f"{API_BASE}/vss/rollback",
                json={"snapshot_id": snapshot_id, "restore_target_path": restore_target_path},
            )
            if res.status_code == 200:
                return res.json()
    except Exception:
        pass

    _, _, vss, _ = _get_local_fallbacks()
    return vss.rollback_snapshot(
        snapshot_id=snapshot_id,
        restore_target_path=restore_target_path,
    )


@mcp.tool()
def restore_workspace_backup(
    snapshot_id: str,
    restore_target_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Restores a folder back to a specific `snapshot_id`."""
    return undo_last_agent_edit(
        snapshot_id=snapshot_id,
        restore_target_path=restore_target_path,
    )


@mcp.tool()
def list_workspace_backups() -> List[Dict[str, Any]]:
    """Lists all manual and automatic workspace backups available for instant rollback."""
    try:
        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{API_BASE}/vss/list")
            if res.status_code == 200:
                return res.json().get("snapshots", [])
    except Exception:
        pass

    _, _, vss, _ = _get_local_fallbacks()
    return vss.list_snapshots()


@mcp.tool()
def summarize_recent_activity(
    minutes: int = 15,
    focus: str = "Summarize the bugs encountered, terminal commands run, and current blocker.",
) -> Dict[str, Any]:
    """Generates an AI root-cause summary of the last N minutes of screen activity."""
    now_ms = int(time.time() * 1000)
    start_ts = now_ms - (minutes * 60 * 1000)
    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.post(
                f"{API_BASE}/ai/summarize",
                json={"start_ts": start_ts, "end_ts": now_ms, "prompt_focus": focus},
            )
            if res.status_code == 200:
                return res.json()
    except Exception:
        pass

    _, _, _, summarizer = _get_local_fallbacks()
    return summarizer.summarize_window(start_ts=start_ts, end_ts=now_ms, prompt_focus=focus)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()