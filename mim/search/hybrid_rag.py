import math
import time
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

from mim.config import CONFIG
from mim.storage.sqlite_manager import SQLiteManager
from mim.storage.lancedb_manager import LanceDBManager

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False


class LocalEmbedder:
    """
    Uses lightweight ONNX Runtime (`fastembed` BAAI/bge-small-en-v1.5, 384-dim)
    to cut RAM usage by ~75% compared to PyTorch, with an instant deterministic
    feature-hashing fallback for offline/cold-start environments.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dim: int = CONFIG.embedding_dim,
    ):
        self.dim = dim
        self.onnx_model = None
        if HAS_FASTEMBED:
            try:
                self.onnx_model = TextEmbedding(model_name=model_name)
            except Exception as exc:
                print(f"[LocalEmbedder] Falling back to fast hash embedder: {exc}")

    def encode(self, text: str) -> List[float]:
        clean = (text or "").strip()
        if not clean:
            return [0.0] * self.dim

        if self.onnx_model is not None:
            try:
                vec = next(iter(self.onnx_model.embed([clean])))
                return np.asarray(vec, dtype=np.float32).tolist()
            except Exception:
                pass

        # Deterministic 384-d token feature hashing fallback (0 MB overhead)
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in clean.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()


class HybridRAGEngine:
    """
    Combines SQLite FTS5 BM25 keyword search and LanceDB ANN vector search using
    Time-Decayed Reciprocal Rank Fusion (TD-RRF).
    """

    def __init__(
        self,
        sqlite_mgr: SQLiteManager,
        lancedb_mgr: LanceDBManager,
        embedder: Optional[LocalEmbedder] = None,
        rrf_k: int = 60,
        decay_lambda: float = 0.015,
    ):
        self.sqlite_mgr = sqlite_mgr
        self.lancedb_mgr = lancedb_mgr
        self.embedder = embedder or LocalEmbedder()
        self.rrf_k = rrf_k
        self.decay_lambda = decay_lambda

    def index_frame_semantics(
        self,
        frame_record: Dict[str, Any],
        ui_nodes: List[Dict[str, Any]],
        flush_now: bool = False,
    ) -> None:
        snippets = [
            frame_record.get("process_name", ""),
            frame_record.get("window_title", ""),
        ]
        for node in ui_nodes[:40]:
            txt = (node.get("content_text") or "").strip()
            if txt:
                snippets.append(txt)

        composite_text = " | ".join(s for s in snippets if s)
        vector = self.embedder.encode(composite_text)

        self.lancedb_mgr.insert_embeddings(
            [{
                "frame_id": frame_record["frame_id"],
                "vector": vector,
                "timestamp_ms": frame_record["timestamp_ms"],
                "process_name": frame_record.get("process_name", "unknown"),
                "semantic_tags": [frame_record.get("process_name", "unknown").lower()],
            }],
            flush_now=flush_now,
        )

    def search(
        self,
        query: str,
        limit: int = 15,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        w_lex: float = 0.55,
        w_vec: float = 0.45,
    ) -> List[Dict[str, Any]]:
        now_ms = int(time.time() * 1000)

        lex_results = self.sqlite_mgr.search_lexical_bm25(
            query=query, limit=limit * 2, start_ts=start_ts, end_ts=end_ts
        )

        query_vec = self.embedder.encode(query)
        try:
            vec_results = self.lancedb_mgr.search_semantic(
                query_vector=query_vec, limit=limit * 2, start_ts=start_ts, end_ts=end_ts
            )
        except Exception:
            vec_results = []

        fused: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(lex_results, start=1):
            fid = item["frame_id"]
            if fid not in fused:
                fused[fid] = {
                    "frame_id": fid,
                    "timestamp_ms": item["timestamp_ms"],
                    "process_name": item["process_name"],
                    "window_title": item["window_title"],
                    "image_path": item["image_path"],
                    "matched_text": item["content_text"],
                    "bounding_box": item["bounding_box"],
                    "lex_rank": rank,
                    "vec_rank": None,
                    "raw_rrf": 0.0,
                }
            fused[fid]["raw_rrf"] += w_lex / (self.rrf_k + rank)

        for rank, item in enumerate(vec_results, start=1):
            fid = item["frame_id"]
            if fid not in fused:
                frame_meta = self.sqlite_mgr.get_frame_at_timestamp(item["timestamp_ms"])
                if not frame_meta:
                    continue
                nodes = self.sqlite_mgr.get_nodes_for_frame(fid)
                first_text = next((n["content_text"] for n in nodes if n.get("content_text")), "")
                first_bbox = nodes[0]["bounding_box"] if nodes else [0, 0, 0, 0]
                fused[fid] = {
                    "frame_id": fid,
                    "timestamp_ms": frame_meta["timestamp_ms"],
                    "process_name": frame_meta["process_name"],
                    "window_title": frame_meta["window_title"],
                    "image_path": frame_meta["image_path"],
                    "matched_text": first_text or frame_meta.get("window_title", ""),
                    "bounding_box": first_bbox,
                    "lex_rank": None,
                    "vec_rank": rank,
                    "raw_rrf": 0.0,
                }
            else:
                if fused[fid]["vec_rank"] is None:
                    fused[fid]["vec_rank"] = rank

            fused[fid]["raw_rrf"] += w_vec / (self.rrf_k + rank)

        ranked_list = []
        for fid, data in fused.items():
            delta_hours = max(0.0, (now_ms - data["timestamp_ms"]) / 3_600_000.0)
            decay_factor = math.exp(-self.decay_lambda * delta_hours)
            data["td_rrf_score"] = round(data["raw_rrf"] * decay_factor, 6)
            ranked_list.append(data)

        ranked_list.sort(key=lambda x: x["td_rrf_score"], reverse=True)
        return ranked_list[:limit]