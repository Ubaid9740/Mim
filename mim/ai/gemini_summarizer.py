import os
from datetime import datetime
from typing import List, Dict, Any
from PIL import Image
from mim.config import BASE_DIR
from mim.storage.sqlite_manager import SQLiteManager

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiSummarizer:
    """
    3-Layer General-Purpose Screen Memory Assistant for `gemini-2.5-flash`:
      1. Multi-Day Hourly Activity Rollup (summarizes days of work without token bloat)
      2. Full-History Keyword Search Hits (finds specific moments from days ago)
      3. Detailed Recent Timeline + 1:1 High-DPI Active-Window Crops
    """

    STOPWORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "have", "has", "had", "been", "working", "since", "last", "days", "day",
        "hours", "hour", "today", "yesterday", "summarize", "summary", "tell",
        "about", "from", "that", "this", "with", "the", "and", "for", "was", "were",
        "explain", "simple", "friendly", "english", "doing", "computer", "apps",
        "websites", "used", "highlight", "important", "text", "tasks", "errors",
    }

    def __init__(self, sqlite_mgr: SQLiteManager, model_name: str = "gemini-2.5-flash"):
        self.sqlite_mgr = sqlite_mgr
        self.model_name = model_name
        try:
            from dotenv import load_dotenv
            load_dotenv(BASE_DIR / ".env", override=True)
        except Exception:
            pass
        self.api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()

    @staticmethod
    def _extract_active_window_crop(
        img: Image.Image,
        nodes: List[Dict[str, Any]],
    ) -> Image.Image:
        if not nodes:
            return img

        root_bbox = nodes[0].get("bounding_box", [0, 0, 0, 0])
        if len(root_bbox) != 4:
            return img

        left, top, right, bottom = root_bbox
        w, h = img.size
        left = max(0, min(w - 1, int(left)))
        top = max(0, min(h - 1, int(top)))
        right = max(left + 1, min(w, int(right)))
        bottom = max(top + 1, min(h, int(bottom)))

        if (right - left) >= 200 and (bottom - top) >= 120:
            return img.crop((left, top, right, bottom))
        return img

    def _build_multiday_hourly_rollup(self, start_ts: int, end_ts: int) -> str:
        """
        Compresses days of screen history into a clean hourly breakdown of apps & window titles
        so Gemini can see 2-7 days of work at a glance.
        """
        conn = self.sqlite_mgr._get_conn()
        rows = conn.execute("""
            SELECT
                (timestamp_ms / 3600000) * 3600000 AS hour_bucket_ms,
                process_name,
                window_title,
                COUNT(*) AS activity_frames
            FROM frames
            WHERE timestamp_ms >= ? AND timestamp_ms <= ?
              AND NOT (LOWER(process_name) = 'python.exe' AND LOWER(window_title) = 'python')
            GROUP BY hour_bucket_ms, process_name, window_title
            ORDER BY hour_bucket_ms ASC, activity_frames DESC
            LIMIT 250
        """, (start_ts, end_ts)).fetchall()

        if not rows:
            return "No multi-day rollup data in range."

        lines = []
        for r in rows:
            dt_str = datetime.fromtimestamp(r["hour_bucket_ms"] / 1000.0).strftime("%b %d, %I:00 %p")
            title = (r["window_title"] or "Untitled").strip()[:80]
            lines.append(
                f"[{dt_str}] App: {r['process_name']} | Window: {title} ({r['activity_frames']} screen updates)"
            )
        return "\n".join(lines)

    def _build_relevant_search_hits(self, prompt_focus: str) -> str:
        """Searches across the entire SQLite FTS5 database for specific keywords in the user's question."""
        cleaned = (
            prompt_focus.lower()
            .replace("?", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace("'", " ")
            .replace('"', " ")
        )
        keywords = [
            w for w in cleaned.split()
            if len(w) >= 3 and w not in self.STOPWORDS
        ]
        if not keywords:
            return ""

        search_q = " ".join(keywords[:8])
        hits = self.sqlite_mgr.search_lexical_bm25(query=search_q, limit=25)
        if not hits:
            return ""

        lines = []
        for h in hits:
            dt_str = datetime.fromtimestamp(h["timestamp_ms"] / 1000.0).strftime("%b %d, %I:%M:%S %p")
            txt = (h.get("content_text") or "").strip()[:220]
            lines.append(
                f"[{dt_str}] App: {h['process_name']} | Window: {h['window_title']} | Matched Text: {txt}"
            )
        return "\n".join(lines)

    def summarize_window(
        self,
        start_ts: int,
        end_ts: int,
        prompt_focus: str = "Summarize the user and AI agent actions, terminal commands, and errors.",
    ) -> Dict[str, Any]:
        conn = self.sqlite_mgr._get_conn()

        total_row = conn.execute("""
            SELECT COUNT(*) FROM frames
            WHERE timestamp_ms >= ? AND timestamp_ms <= ?
              AND NOT (LOWER(process_name) = 'python.exe' AND LOWER(window_title) = 'python')
        """, (start_ts, end_ts)).fetchone()
        total_in_window = total_row[0] if total_row else 0

        frames = conn.execute("""
            SELECT * FROM frames
            WHERE timestamp_ms >= ? AND timestamp_ms <= ?
              AND NOT (LOWER(process_name) = 'python.exe' AND LOWER(window_title) = 'python')
            ORDER BY timestamp_ms DESC
            LIMIT 120
        """, (start_ts, end_ts)).fetchall()

        if not frames:
            frames = conn.execute("""
                SELECT * FROM frames
                WHERE NOT (LOWER(process_name) = 'python.exe' AND LOWER(window_title) = 'python')
                ORDER BY timestamp_ms DESC
                LIMIT 60
            """).fetchall()
            total_in_window = len(frames)

        if not frames:
            return {
                "engine": "none",
                "frame_count": 0,
                "summary": "No screen history recorded yet.",
            }

        frames = list(reversed(frames))
        timeline_log: List[str] = []
        pil_images: List[Image.Image] = []

        newest_for_images = frames[-6:]
        newest_ids = {f["frame_id"] for f in newest_for_images}

        for f in frames:
            fid = f["frame_id"]
            nodes = self.sqlite_mgr.get_nodes_for_frame(fid)
            texts = [n["content_text"] for n in nodes if (n.get("content_text") or "").strip()]
            snippet = " | ".join(texts[:10])
            human_time = datetime.fromtimestamp(f["timestamp_ms"] / 1000.0).strftime("%b %d, %I:%M:%S %p")
            timeline_log.append(
                f"[{human_time}] App: {f['process_name']} | Window: {f['window_title']} | Text: {snippet[:300]}"
            )

            if fid in newest_ids and len(pil_images) < 6:
                img_file = BASE_DIR / f["image_path"]
                if img_file.exists():
                    try:
                        full_im = Image.open(img_file).copy()
                        roi_crop = self._extract_active_window_crop(full_im, nodes)
                        pil_images.append(roi_crop)
                    except Exception:
                        pass

        hourly_rollup = self._build_multiday_hourly_rollup(start_ts, end_ts)
        keyword_hits = self._build_relevant_search_hits(prompt_focus)
        compiled_transcript = "\n".join(timeline_log)

        if HAS_GENAI and self.api_key:
            try:
                client = genai.Client(api_key=self.api_key)
                prompt_body = (
                    "You are Mim, an intelligent desktop memory assistant.\n"
                    f"User Request / Question: {prompt_focus}\n\n"
                    "=== LAYER 1: MULTI-DAY HOURLY ACTIVITY ROLLUP (All Apps & Windows Over Time) ===\n"
                    f"{hourly_rollup}\n\n"
                )
                if keyword_hits:
                    prompt_body += (
                        "=== LAYER 2: EXACT SEARCH MATCHES FROM ENTIRE DATABASE ===\n"
                        f"{keyword_hits}\n\n"
                    )
                prompt_body += (
                    "=== LAYER 3: DETAILED RECENT SCREEN TEXT LOG ===\n"
                    f"{compiled_transcript}"
                )

                contents: List[Any] = [prompt_body]
                contents.extend(pil_images)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return {
                    "engine": self.model_name,
                    "frame_count": max(total_in_window, len(frames)),
                    "roi_crops_attached": len(pil_images),
                    "summary": response.text,
                }
            except Exception as exc:
                print(f"[GeminiSummarizer] API fallback triggered: {exc}")

        apps = sorted({f["process_name"] for f in frames})
        return {
            "engine": "local-heuristic-fallback",
            "frame_count": max(total_in_window, len(frames)),
            "roi_crops_attached": len(pil_images),
            "apps_observed": apps,
            "summary": (
                f"Session covered {max(total_in_window, len(frames))} delta frames across {', '.join(apps)}.\n\n"
                f"Hourly Rollup:\n{hourly_rollup[:1500]}"
            ),
        }