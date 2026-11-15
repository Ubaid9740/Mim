import sqlite3
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from mim.config import CONFIG


class SQLiteManager:
    """
    Thread-safe SQLite manager with WAL mode, auto-migrating FTS5 BM25 lexical search,
    Hot/Warm image tiering, and Cold Milestone archival pruning.
    """

    def __init__(self, db_path: Path = CONFIG.sqlite_path):
        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._compressed_warm_ids: set[str] = set()
        self._pruned_cold_ids: set[str] = set()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA mmap_size=67108864;")  # 64 MB cap for low RAM footprint
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    timestamp_ms INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    dhash TEXT NOT NULL,
                    active_hwnd INTEGER NOT NULL,
                    process_name TEXT NOT NULL,
                    window_title TEXT,
                    vss_snapshot_id TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_frames_timestamp
                ON frames(timestamp_ms DESC);

                CREATE INDEX IF NOT EXISTS idx_frames_process
                ON frames(process_name, timestamp_ms DESC);

                CREATE TABLE IF NOT EXISTS ui_nodes (
                    node_id TEXT PRIMARY KEY,
                    frame_id TEXT NOT NULL,
                    parent_node_id TEXT,
                    control_type INTEGER NOT NULL,
                    bounding_box TEXT NOT NULL,
                    is_interactable INTEGER NOT NULL DEFAULT 1,
                    content_text TEXT,
                    source_type TEXT NOT NULL DEFAULT 'UIA',
                    FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_uinodes_frame
                ON ui_nodes(frame_id);

                CREATE TABLE IF NOT EXISTS network_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp_ms INTEGER NOT NULL,
                    pid INTEGER NOT NULL,
                    process_name TEXT NOT NULL,
                    protocol TEXT NOT NULL,
                    dest_ip TEXT NOT NULL,
                    dest_port INTEGER NOT NULL,
                    payload_size INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_net_timestamp
                ON network_events(timestamp_ms DESC);
            """)

            # Check if `fts_ui_text` exists and has the current 5-column schema
            fts_cols = [
                row["name"]
                for row in cursor.execute("PRAGMA table_info(fts_ui_text);").fetchall()
            ]
            required_cols = {"node_id", "frame_id", "process_name", "window_title", "content_text"}

            if not required_cols.issubset(set(fts_cols)):
                # Self-heal legacy FTS5 table and backfill from existing frames + ui_nodes
                cursor.executescript("""
                    DROP TABLE IF EXISTS fts_ui_text;
                    CREATE VIRTUAL TABLE fts_ui_text USING fts5(
                        node_id UNINDEXED,
                        frame_id UNINDEXED,
                        process_name,
                        window_title,
                        content_text,
                        tokenize = 'porter unicode61'
                    );
                    INSERT INTO fts_ui_text (node_id, frame_id, process_name, window_title, content_text)
                    SELECT u.node_id, u.frame_id, f.process_name, COALESCE(f.window_title, ''), u.content_text
                    FROM ui_nodes u
                    JOIN frames f ON f.frame_id = u.frame_id
                    WHERE u.content_text IS NOT NULL AND TRIM(u.content_text) != '';
                """)

            conn.commit()

    def insert_frame_bundle(
        self,
        frame_data: Dict[str, Any],
        ui_nodes: List[Dict[str, Any]]
    ) -> None:
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
                cursor.execute("""
                    INSERT INTO frames (
                        frame_id, timestamp_ms, image_path, dhash,
                        active_hwnd, process_name, window_title, vss_snapshot_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    frame_data["frame_id"],
                    frame_data["timestamp_ms"],
                    frame_data["image_path"],
                    frame_data["dhash"],
                    frame_data["active_hwnd"],
                    frame_data["process_name"],
                    frame_data.get("window_title", ""),
                    frame_data.get("vss_snapshot_id"),
                ))

                node_rows = []
                fts_rows = []
                for n in ui_nodes:
                    bbox_json = json.dumps(n["bounding_box"])
                    text_val = (n.get("content_text") or "").strip()
                    node_rows.append((
                        n["node_id"],
                        frame_data["frame_id"],
                        n.get("parent_node_id"),
                        n["control_type"],
                        bbox_json,
                        1 if n.get("is_interactable", True) else 0,
                        text_val,
                        n.get("source_type", "UIA"),
                    ))
                    if text_val:
                        fts_rows.append((
                            n["node_id"],
                            frame_data["frame_id"],
                            frame_data["process_name"],
                            frame_data.get("window_title", ""),
                            text_val,
                        ))

                if node_rows:
                    cursor.executemany("""
                        INSERT INTO ui_nodes (
                            node_id, frame_id, parent_node_id, control_type,
                            bounding_box, is_interactable, content_text, source_type
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, node_rows)

                if fts_rows:
                    cursor.executemany("""
                        INSERT INTO fts_ui_text (
                            node_id, frame_id, process_name, window_title, content_text
                        ) VALUES (?, ?, ?, ?, ?)
                    """, fts_rows)

                conn.commit()
            except Exception as exc:
                conn.rollback()
                raise RuntimeError(f"Failed to commit frame bundle: {exc}") from exc

    def insert_network_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            rows = [
                (
                    e["event_id"],
                    e["timestamp_ms"],
                    e["pid"],
                    e["process_name"],
                    e["protocol"],
                    e["dest_ip"],
                    e["dest_port"],
                    e.get("payload_size", 0),
                )
                for e in events
            ]
            cursor.executemany("""
                INSERT OR IGNORE INTO network_events (
                    event_id, timestamp_ms, pid, process_name,
                    protocol, dest_ip, dest_port, payload_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    def search_lexical_bm25(
        self,
        query: str,
        limit: int = 20,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        safe_tokens = []
        for tok in query.strip().split():
            # Strip punctuation that could confuse FTS5 syntax
            clean_tok = "".join(ch for ch in tok if ch.isalnum() or ch in ("_", "-"))
            if clean_tok:
                safe_tokens.append(f'"{clean_tok}"')

        if not safe_tokens:
            return []
        fts_query = " OR ".join(safe_tokens)

        sql = """
            SELECT
                fts.node_id,
                fts.frame_id,
                f.process_name,
                f.window_title,
                fts.content_text,
                f.timestamp_ms,
                f.image_path,
                u.bounding_box,
                bm25(fts_ui_text) AS bm25_score
            FROM fts_ui_text fts
            JOIN frames f ON f.frame_id = fts.frame_id
            JOIN ui_nodes u ON u.node_id = fts.node_id
            WHERE fts_ui_text MATCH ?
        """
        params: List[Any] = [fts_query]

        if start_ts is not None:
            sql += " AND f.timestamp_ms >= ?"
            params.append(start_ts)
        if end_ts is not None:
            sql += " AND f.timestamp_ms <= ?"
            params.append(end_ts)

        sql += " ORDER BY bm25_score ASC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            item = dict(r)
            item["bounding_box"] = json.loads(item["bounding_box"])
            results.append(item)
        return results

    def get_frame_at_timestamp(self, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute("""
            SELECT * FROM frames
            ORDER BY ABS(timestamp_ms - ?) ASC
            LIMIT 1
        """, (timestamp_ms,)).fetchone()
        return dict(row) if row else None

    def get_nodes_for_frame(self, frame_id: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM ui_nodes WHERE frame_id = ?
        """, (frame_id,)).fetchall()
        nodes = []
        for r in rows:
            d = dict(r)
            d["bounding_box"] = json.loads(d["bounding_box"])
            d["is_interactable"] = bool(d["is_interactable"])
            nodes.append(d)
        return nodes

    def compress_warm_tier_frames(
        self,
        hot_keep: int = CONFIG.hot_tier_frames,
        warm_quality: int = CONFIG.warm_tier_quality,
        base_dir: Path = CONFIG.data_dir.parent,
    ) -> int:
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT frame_id, image_path FROM frames
            ORDER BY timestamp_ms DESC
            LIMIT 500 OFFSET ?
        """, (hot_keep,)).fetchall()

        compressed = 0
        for r in rows:
            fid = r["frame_id"]
            if fid in self._compressed_warm_ids:
                continue
            img_path = base_dir / r["image_path"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as im:
                        im.save(img_path, format="WEBP", quality=warm_quality, method=4)
                    compressed += 1
                except Exception:
                    pass
            self._compressed_warm_ids.add(fid)
        return compressed

    def prune_old_frames(
        self,
        max_frames: int = CONFIG.max_stored_frames,
        base_dir: Path = CONFIG.data_dir.parent,
        lancedb_mgr: Optional[Any] = None,
        hard_delete_db_rows: Optional[bool] = None,
    ) -> int:
        if hard_delete_db_rows is None:
            hard_delete_db_rows = (max_frames < 100)

        conn = self._get_conn()
        with self._write_lock:
            total_row = conn.execute("SELECT COUNT(*) FROM frames").fetchone()
            total = total_row[0] if total_row else 0
            if total <= max_frames:
                return 0

            excess = total - max_frames
            old_rows = conn.execute("""
                SELECT frame_id, image_path, timestamp_ms, process_name, window_title, vss_snapshot_id
                FROM frames
                ORDER BY timestamp_ms ASC
                LIMIT ?
            """, (excess,)).fetchall()

            if not old_rows:
                return 0

            if hard_delete_db_rows:
                frame_ids = [r["frame_id"] for r in old_rows]
                cutoff_ts = old_rows[-1]["timestamp_ms"]
                placeholders = ",".join("?" for _ in frame_ids)

                cursor = conn.cursor()
                try:
                    cursor.execute("BEGIN IMMEDIATE TRANSACTION;")
                    cursor.execute(
                        f"DELETE FROM fts_ui_text WHERE frame_id IN ({placeholders})",
                        frame_ids,
                    )
                    cursor.execute(
                        f"DELETE FROM frames WHERE frame_id IN ({placeholders})",
                        frame_ids,
                    )
                    cursor.execute(
                        "DELETE FROM network_events WHERE timestamp_ms <= ?",
                        (cutoff_ts,),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    return 0

                for r in old_rows:
                    self._compressed_warm_ids.discard(r["frame_id"])
                    img_file = base_dir / r["image_path"]
                    if img_file.exists():
                        try:
                            img_file.unlink()
                        except Exception:
                            pass

                if lancedb_mgr is not None and hasattr(lancedb_mgr, "delete_frames"):
                    try:
                        lancedb_mgr.delete_frames(frame_ids)
                    except Exception:
                        pass

                return len(frame_ids)

        pruned_images = 0
        last_session_sig = None

        for r in old_rows:
            fid = r["frame_id"]
            session_sig = (
                (r["process_name"] or "").lower(),
                (r["window_title"] or "").strip().lower(),
            )
            has_backup = bool(r["vss_snapshot_id"])

            is_milestone = (session_sig != last_session_sig) or has_backup
            last_session_sig = session_sig

            if fid in self._pruned_cold_ids:
                continue

            img_file = base_dir / r["image_path"]
            if is_milestone:
                if img_file.exists():
                    try:
                        with Image.open(img_file) as im:
                            im.save(img_file, format="WEBP", quality=38, method=4)
                    except Exception:
                        pass
                self._pruned_cold_ids.add(fid)
            else:
                if img_file.exists():
                    try:
                        img_file.unlink()
                        pruned_images += 1
                    except Exception:
                        pass
                self._pruned_cold_ids.add(fid)

        return pruned_images

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            except Exception:
                pass
            self._local.conn.close()
            self._local.conn = None