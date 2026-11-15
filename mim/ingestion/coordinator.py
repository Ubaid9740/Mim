import os
import time
import uuid
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import psutil
from PIL import Image, ImageDraw
import uiautomation as auto

from mim.config import CONFIG, BASE_DIR
from mim.storage.sqlite_manager import SQLiteManager
from mim.storage.lancedb_manager import LanceDBManager
from mim.security.process_guard import ProcessGuard
from mim.security.sanitizer import ZeroTrustSanitizer
from mim.ingestion.vision_grabber import VisionGrabber
from mim.ingestion.uia_scraper import UIAScraper
from mim.system.vss_manager import VSSManager


class SentinelState(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED_MANUAL = "PAUSED_MANUAL"
    PAUSED_BLACKLIST = "PAUSED_BLACKLIST"


class IngestionCoordinator:
    """
    Orchestrates 2-FPS 100% Native Delta Vision, Throttled UIA Scraping,
    Pixel Blackout Redaction, Hot/Warm Image Tiering, Synced SQLite+LanceDB Pruning,
    and Dirty-Aware Auto-Backups.
    """

    def __init__(
        self,
        sqlite_mgr: SQLiteManager,
        lancedb_mgr: Optional[LanceDBManager] = None,
        vss_mgr: Optional[VSSManager] = None,
        process_guard: Optional[ProcessGuard] = None,
        sanitizer: Optional[ZeroTrustSanitizer] = None,
        on_frame_committed: Optional[Callable[[Dict[str, Any], int], None]] = None,
        watched_workspace: Optional[str] = None,
    ):
        self.sqlite_mgr = sqlite_mgr
        self.lancedb_mgr = lancedb_mgr
        self.vss_mgr = vss_mgr or VSSManager()
        self.process_guard = process_guard or ProcessGuard()
        self.sanitizer = sanitizer or ZeroTrustSanitizer()
        self.vision = VisionGrabber()
        self.uia = UIAScraper(sanitizer=self.sanitizer)
        self.on_frame_committed = on_frame_committed
        self.watched_workspace = watched_workspace or str(BASE_DIR)
        self._own_pid = os.getpid()

        self._state_lock = threading.Lock()
        self._state: SentinelState = SentinelState.ACTIVE
        self._paused_pid: Optional[int] = None
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._last_active_hwnd: int = 0
        self._last_uia_scrape_ts: float = 0.0
        self._cached_uia_nodes: List[Dict[str, Any]] = []
        self._last_auto_backup_ts: float = 0.0

        self.stats = {
            "ticks_total": 0,
            "frames_committed": 0,
            "frames_skipped_static": 0,
            "frames_blocked_security": 0,
            "uia_nodes_indexed": 0,
            "uia_walks_throttled": 0,
            "pixel_regions_redacted": 0,
            "frames_pruned": 0,
            "warm_frames_compressed": 0,
            "auto_backups_created": 0,
            "auto_backups_skipped_clean": 0,
        }

    @property
    def state(self) -> SentinelState:
        with self._state_lock:
            return self._state

    def toggle_manual_pause(self, bind_to_active_pid: bool = True) -> Dict[str, Any]:
        with self._state_lock:
            if self._state in (SentinelState.ACTIVE, SentinelState.PAUSED_BLACKLIST):
                self._state = SentinelState.PAUSED_MANUAL
                if bind_to_active_pid:
                    _, pid, _, _ = self.process_guard.get_active_window_metadata()
                    self._paused_pid = pid if (pid > 0 and pid != self._own_pid) else None
                else:
                    self._paused_pid = None
            else:
                self._state = SentinelState.ACTIVE
                self._paused_pid = None
                self.vision.reset_baseline()

            return {
                "state": self._state.value,
                "paused_pid": self._paused_pid,
                "stats": dict(self.stats),
            }

    def _check_auto_wake_from_manual_pause(self) -> bool:
        with self._state_lock:
            if self._state != SentinelState.PAUSED_MANUAL:
                return False
            if self._paused_pid is not None and not psutil.pid_exists(self._paused_pid):
                self._state = SentinelState.ACTIVE
                self._paused_pid = None
                self.vision.reset_baseline()
                return False
            return True

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._ingestion_loop,
            name="MimIngestionWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        if self.lancedb_mgr is not None:
            try:
                self.lancedb_mgr.flush()
            except Exception:
                pass
        self.vision.close()

    def _ingestion_loop(self) -> None:
        interval = 1.0 / max(CONFIG.capture_fps, 0.1)
        with auto.UIAutomationInitializerInThread():
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()
                try:
                    self._execute_tick()
                except Exception as exc:
                    print(f"[MimCoordinator] Tick warning: {exc}")

                elapsed = time.perf_counter() - loop_start
                sleep_duration = max(0.0, interval - elapsed)
                self._stop_event.wait(sleep_duration)

    def _clone_cached_nodes_with_new_ids(self) -> List[Dict[str, Any]]:
        cloned = []
        for n in self._cached_uia_nodes:
            cloned.append({
                "node_id": str(uuid.uuid4()),
                "parent_node_id": None,
                "control_type": n["control_type"],
                "bounding_box": n["bounding_box"],
                "is_interactable": n["is_interactable"],
                "content_text": n["content_text"],
                "source_type": n["source_type"],
            })
        return cloned

    @staticmethod
    def apply_pixel_blackout_redaction(
        img_file: Path,
        ui_nodes: List[Dict[str, Any]],
        quality: int = CONFIG.jpeg_quality,
    ) -> int:
        """
        Scans `ui_nodes` for any text where ZeroTrustSanitizer injected a `[REDACTED` token.
        Paints a solid black rectangle over those exact bounding box coordinates on `img_file`
        so secrets are permanently erased from the `.webp` pixels on disk.
        """
        redacted_boxes = []
        for node in ui_nodes:
            txt = node.get("content_text") or ""
            if "[REDACTED" in txt:
                bbox = node.get("bounding_box", [0, 0, 0, 0])
                if len(bbox) == 4:
                    l, t, r, b = bbox
                    if (r - l) > 2 and (b - t) > 2:
                        redacted_boxes.append((int(l), int(t), int(r), int(b)))

        if not redacted_boxes or not img_file.exists():
            return 0

        try:
            with Image.open(img_file) as im:
                im_copy = im.convert("RGB")
                draw = ImageDraw.Draw(im_copy)
                w, h = im_copy.size
                count = 0
                for l, t, r, b in redacted_boxes:
                    cl = max(0, min(w - 1, l))
                    ct = max(0, min(h - 1, t))
                    cr = max(cl + 1, min(w, r))
                    cb = max(ct + 1, min(h, b))
                    # Avoid blacking out the entire root window container (>75% of screen area)
                    if (cr - cl) * (cb - ct) < (w * h * 0.75):
                        draw.rectangle([cl, ct, cr, cb], fill=(0, 0, 0), outline=(255, 59, 48), width=1)
                        count += 1
                if count > 0:
                    im_copy.save(img_file, format="WEBP", quality=quality, method=4)
                return count
        except Exception:
            return 0

    def _execute_tick(self) -> None:
        self.stats["ticks_total"] += 1

        if self._check_auto_wake_from_manual_pause():
            return

        is_blocked, hwnd, pid, proc_name, window_title = self.process_guard.inspect_foreground()

        # Ignore Mim's own floating Eye Sentinel and Scrubber Overlay windows
        if pid == self._own_pid or (proc_name.lower() == "python.exe" and window_title.lower() in ("python", "mim")):
            return

        if is_blocked:
            with self._state_lock:
                self._state = SentinelState.PAUSED_BLACKLIST
            self.stats["frames_blocked_security"] += 1
            self.vision.reset_baseline()
            return
        else:
            with self._state_lock:
                if self._state == SentinelState.PAUSED_BLACKLIST:
                    self._state = SentinelState.ACTIVE

        hwnd_switched = (hwnd != 0 and hwnd != self._last_active_hwnd)
        frame_id = str(uuid.uuid4())
        now_sec = time.time()
        timestamp_ms = int(now_sec * 1000)

        capture_res = self.vision.capture_delta_frame(
            frame_id=frame_id,
            force=hwnd_switched,
        )
        if capture_res is None:
            self.stats["frames_skipped_static"] += 1
            return

        image_path, dhash_hex, _ = capture_res
        self._last_active_hwnd = hwnd

        sanitized_title = self.sanitizer.sanitize(window_title, process_name=proc_name)
        if hwnd_switched or (now_sec - self._last_uia_scrape_ts) >= CONFIG.uia_scrape_interval_sec:
            ui_nodes = self.uia.scrape_window(hwnd=hwnd, process_name=proc_name)
            self._cached_uia_nodes = ui_nodes
            self._last_uia_scrape_ts = now_sec
        else:
            ui_nodes = self._clone_cached_nodes_with_new_ids()
            self.stats["uia_walks_throttled"] += 1

        # Burn solid black censor boxes onto the .webp screenshot over any [REDACTED] secret coordinates
        full_img_path = BASE_DIR / image_path
        redacted_count = self.apply_pixel_blackout_redaction(full_img_path, ui_nodes)
        self.stats["pixel_regions_redacted"] += redacted_count

        auto_snap_id = None
        if (now_sec - self._last_auto_backup_ts) >= CONFIG.auto_backup_interval_sec:
            self._last_auto_backup_ts = now_sec
            try:
                snap_meta = self.vss_mgr.create_snapshot(
                    target_path=self.watched_workspace,
                    is_auto=True,
                    skip_if_unchanged=True,
                )
                if snap_meta.get("status") == "SKIPPED_UNCHANGED":
                    self.stats["auto_backups_skipped_clean"] += 1
                else:
                    auto_snap_id = snap_meta["snapshot_id"]
                    self.stats["auto_backups_created"] += 1
            except Exception:
                pass

        frame_record = {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "image_path": image_path,
            "dhash": dhash_hex,
            "active_hwnd": hwnd,
            "process_name": proc_name,
            "window_title": sanitized_title,
            "vss_snapshot_id": auto_snap_id,
        }

        self.sqlite_mgr.insert_frame_bundle(frame_record, ui_nodes)
        self.stats["frames_committed"] += 1
        self.stats["uia_nodes_indexed"] += len(ui_nodes)

        if self.stats["frames_committed"] % 50 == 0:
            warmed = self.sqlite_mgr.compress_warm_tier_frames(hot_keep=CONFIG.hot_tier_frames)
            pruned = self.sqlite_mgr.prune_old_frames(
                max_frames=CONFIG.max_stored_frames,
                lancedb_mgr=self.lancedb_mgr,
            )
            self.stats["warm_frames_compressed"] += warmed
            self.stats["frames_pruned"] += pruned

        if self.on_frame_committed:
            try:
                self.on_frame_committed(frame_record, len(ui_nodes))
            except Exception:
                pass