import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import lancedb
import pyarrow as pa
from mim.config import CONFIG


class LanceDBManager:
    """
    Manages the embedded LanceDB vector store for 384-d semantic frame embeddings.
    Includes in-memory write batching to prevent SSD fragmentation and synced deletion.
    """

    TABLE_NAME = "frame_vectors"

    def __init__(
        self,
        uri: Path = CONFIG.lancedb_uri,
        embedding_dim: int = CONFIG.embedding_dim,
        batch_size: int = CONFIG.lancedb_batch_size,
    ):
        self.uri = uri
        self.embedding_dim = embedding_dim
        self.batch_size = max(1, batch_size)
        self._lock = threading.Lock()
        self._buffer: List[Dict[str, Any]] = []

        self.db = lancedb.connect(str(self.uri))
        self.schema = pa.schema([
            pa.field("frame_id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_dim), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
            pa.field("process_name", pa.string(), nullable=False),
            pa.field("semantic_tags", pa.list_(pa.string()), nullable=True),
        ])
        self._init_table()

    def _init_table(self) -> None:
        try:
            existing = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
        except Exception:
            existing = []

        if self.TABLE_NAME in existing:
            self.table = self.db.open_table(self.TABLE_NAME)
        else:
            self.table = self.db.create_table(self.TABLE_NAME, schema=self.schema)

    def insert_embeddings(self, records: List[Dict[str, Any]], flush_now: bool = False) -> None:
        """
        Buffers vector records in RAM and writes to LanceDB only when `batch_size` is reached
        (or when `flush_now=True`), eliminating per-frame SSD write amplification.
        """
        if not records:
            return
        with self._lock:
            self._buffer.extend(records)
            if flush_now or len(self._buffer) >= self.batch_size:
                self._flush_unlocked()

    def flush(self) -> int:
        """Flushes any pending buffered vectors to disk immediately."""
        with self._lock:
            return self._flush_unlocked()

    def _flush_unlocked(self) -> int:
        if not self._buffer:
            return 0
        to_write = list(self._buffer)
        self._buffer.clear()
        self.table.add(to_write)
        return len(to_write)

    def delete_frames(self, frame_ids: List[str]) -> int:
        """Purges deleted frame vectors from both the RAM buffer and LanceDB disk table."""
        if not frame_ids:
            return 0
        id_set = set(frame_ids)
        with self._lock:
            self._buffer = [r for r in self._buffer if r.get("frame_id") not in id_set]
            quoted = ", ".join(f"'{fid}'" for fid in frame_ids)
            try:
                self.table.delete(f"frame_id IN ({quoted})")
            except Exception:
                pass
        return len(frame_ids)

    def search_semantic(
        self,
        query_vector: List[float],
        limit: int = 20,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        process_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Flushes pending buffer so newest frames are searchable, then runs ANN search."""
        self.flush()

        q = self.table.search(query_vector).limit(limit)
        filters = []
        if start_ts is not None:
            filters.append(f"timestamp_ms >= {int(start_ts)}")
        if end_ts is not None:
            filters.append(f"timestamp_ms <= {int(end_ts)}")
        if process_filter:
            safe_proc = process_filter.replace("'", "''")
            filters.append(f"process_name = '{safe_proc}'")

        if filters:
            q = q.where(" AND ".join(filters))

        return q.to_list()