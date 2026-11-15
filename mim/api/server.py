import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mim.config import CONFIG, BASE_DIR
from mim.storage.sqlite_manager import SQLiteManager
from mim.storage.lancedb_manager import LanceDBManager
from mim.ingestion.coordinator import IngestionCoordinator
from mim.search.hybrid_rag import HybridRAGEngine
from mim.system.vss_manager import VSSManager
from mim.system.net_tracer import NetworkTracer
from mim.ai.gemini_summarizer import GeminiSummarizer


class SnapshotRequest(BaseModel):
    target_path: Optional[str] = None
    frame_id: Optional[str] = None
    is_auto: bool = False
    skip_if_unchanged: bool = False


class RollbackRequest(BaseModel):
    snapshot_id: Optional[str] = None
    restore_target_path: Optional[str] = None


class SummarizeRequest(BaseModel):
    start_ts: int
    end_ts: int
    prompt_focus: str = "Summarize the debugging steps, commands executed, and root cause of any errors."


def create_mim_app(
    sqlite_mgr: Optional[SQLiteManager] = None,
    lancedb_mgr: Optional[LanceDBManager] = None,
    coordinator: Optional[IngestionCoordinator] = None,
    rag_engine: Optional[HybridRAGEngine] = None,
    vss_mgr: Optional[VSSManager] = None,
    net_tracer: Optional[NetworkTracer] = None,
) -> FastAPI:
    """Factory that constructs the FastAPI Agent Bridge and binds all Mim subsystems."""
    app = FastAPI(
        title="Mim (Mimir) Local Context & Flight Recorder API",
        description="Zero-Copy Temporal Desktop Memory Substrate & AI Observability Bridge",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sql = sqlite_mgr or SQLiteManager()
    lance = lancedb_mgr or LanceDBManager()
    rag = rag_engine or HybridRAGEngine(sqlite_mgr=sql, lancedb_mgr=lance)
    vss = vss_mgr or VSSManager()
    coord = coordinator or IngestionCoordinator(
        sqlite_mgr=sql,
        lancedb_mgr=lance,
        vss_mgr=vss,
        on_frame_committed=lambda frame, _: None,
    )
    tracer = net_tracer or NetworkTracer(sqlite_mgr=sql, poll_interval=3.0)
    summarizer = GeminiSummarizer(sqlite_mgr=sql)

    @app.get("/health")
    def health_check():
        return {
            "status": "ok",
            "sentinel_state": coord.state.value,
            "stats": coord.stats,
            "network_events_logged": tracer.total_logged,
        }

    @app.get("/timeline")
    def get_timeline(
        start_ts: Optional[int] = Query(None),
        end_ts: Optional[int] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ):
        conn = sql._get_conn()
        now_ms = int(time.time() * 1000)
        s_ts = start_ts if start_ts is not None else (now_ms - 3_600_000)
        e_ts = end_ts if end_ts is not None else (now_ms + 60_000)

        frames = conn.execute("""
            SELECT * FROM frames
            WHERE timestamp_ms >= ? AND timestamp_ms <= ?
            ORDER BY timestamp_ms DESC
            LIMIT ?
        """, (s_ts, e_ts, limit)).fetchall()

        net_events = conn.execute("""
            SELECT * FROM network_events
            WHERE timestamp_ms >= ? AND timestamp_ms <= ?
            ORDER BY timestamp_ms DESC
            LIMIT ?
        """, (s_ts, e_ts, min(limit, 200))).fetchall()

        return {
            "frames": [dict(f) for f in frames],
            "network_events": [dict(n) for n in net_events],
        }

    @app.get("/frame/{frame_id}/nodes")
    def get_frame_nodes(frame_id: str):
        nodes = sql.get_nodes_for_frame(frame_id)
        return {"frame_id": frame_id, "count": len(nodes), "nodes": nodes}

    @app.get("/frame/{frame_id}/image")
    def get_frame_image(frame_id: str):
        conn = sql._get_conn()
        row = conn.execute(
            "SELECT image_path FROM frames WHERE frame_id = ?", (frame_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Frame ID not found in database.")

        img_path = BASE_DIR / row["image_path"]
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Frame image file missing on disk.")

        return FileResponse(str(img_path), media_type="image/webp")

    @app.get("/search")
    def hybrid_search(
        q: str = Query(..., min_length=1),
        limit: int = Query(15, ge=1, le=100),
        start_ts: Optional[int] = Query(None),
        end_ts: Optional[int] = Query(None),
    ):
        hits = rag.search(query=q, limit=limit, start_ts=start_ts, end_ts=end_ts)
        return {"query": q, "count": len(hits), "results": hits}

    @app.get("/vss/list")
    def list_vss_snapshots():
        snaps = vss.list_snapshots()
        return {"count": len(snaps), "snapshots": snaps}

    @app.post("/vss/snapshot")
    def trigger_vss_snapshot(req: SnapshotRequest):
        meta = vss.create_snapshot(
            target_path=req.target_path,
            is_auto=req.is_auto,
            skip_if_unchanged=req.skip_if_unchanged,
        )
        if req.frame_id and meta.get("snapshot_id"):
            conn = sql._get_conn()
            with sql._write_lock:
                conn.execute(
                    "UPDATE frames SET vss_snapshot_id = ? WHERE frame_id = ?",
                    (meta["snapshot_id"], req.frame_id),
                )
                conn.commit()
        return meta

    @app.post("/vss/rollback")
    def execute_vss_rollback(req: RollbackRequest):
        try:
            return vss.rollback_snapshot(
                snapshot_id=req.snapshot_id,
                restore_target_path=req.restore_target_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/sentinel/toggle")
    def toggle_eye_sentinel(bind_to_active_pid: bool = True):
        return coord.toggle_manual_pause(bind_to_active_pid=bind_to_active_pid)

    @app.post("/ai/summarize")
    def summarize_session(req: SummarizeRequest):
        return summarizer.summarize_window(
            start_ts=req.start_ts,
            end_ts=req.end_ts,
            prompt_focus=req.prompt_focus,
        )

    return app


app = create_mim_app()