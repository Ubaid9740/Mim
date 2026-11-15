import os
import json
import time
import uuid
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
from mim.config import CONFIG, BASE_DIR


class VSSManager:
    """
    Developer Workspace Backup & Agentic Undo Manager.
    Features Dirty-Aware Auto-Backups (skips zipping if no files changed)
    and rolling retention capped at `max_auto_backups`.
    """

    IGNORE_DIRS: Set[str] = {
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".pytest_cache", "data", "dist", "build", ".idea", ".vscode"
    }

    def __init__(self, snapshots_dir: Optional[Path] = None):
        self.snapshots_dir = snapshots_dir or (CONFIG.data_dir / "snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.snapshots_dir / "registry.json"
        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_registry(self) -> None:
        try:
            self.registry_file.write_text(
                json.dumps(self._registry, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _get_latest_mtime_ns(self, folder: Path) -> int:
        """Scans `folder` (excluding IGNORE_DIRS) and returns the newest file modification time in ns."""
        if folder.is_file():
            return folder.stat().st_mtime_ns

        latest_ns = 0
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
            for file in files:
                if file.endswith((".pyc", ".db", ".db-wal", ".db-shm", ".webp")):
                    continue
                full_p = Path(root) / file
                try:
                    mtime = full_p.stat().st_mtime_ns
                    if mtime > latest_ns:
                        latest_ns = mtime
                except Exception:
                    pass
        return latest_ns

    def _prune_auto_backups(self, max_keep: int = CONFIG.max_auto_backups) -> int:
        auto_entries = [
            meta for meta in self._registry.values()
            if meta.get("is_auto", False)
        ]
        if len(auto_entries) <= max_keep:
            return 0

        auto_entries.sort(key=lambda x: x.get("timestamp_ms", 0))
        to_remove = auto_entries[: len(auto_entries) - max_keep]
        removed = 0

        for meta in to_remove:
            sid = meta["snapshot_id"]
            zip_path = Path(meta.get("archive_path", ""))
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            self._registry.pop(sid, None)
            removed += 1

        self._save_registry()
        return removed

    def create_snapshot(
        self,
        target_path: Optional[str] = None,
        volume: str = "C:\\",
        is_auto: bool = False,
        skip_if_unchanged: bool = False,
    ) -> Dict[str, Any]:
        """
        Creates a fast ZIP backup of `target_path`.
        If `skip_if_unchanged=True` and no file has been modified since the last backup
        of that same path, returns the existing snapshot without writing to disk.
        """
        self._registry = self._load_registry()
        resolved_target = Path(target_path).resolve() if target_path else BASE_DIR
        if not resolved_target.exists():
            resolved_target = BASE_DIR

        latest_mtime_ns = self._get_latest_mtime_ns(resolved_target)

        if skip_if_unchanged:
            existing_for_target = [
                m for m in self._registry.values()
                if m.get("target_path") == str(resolved_target)
            ]
            if existing_for_target:
                existing_for_target.sort(key=lambda x: x.get("timestamp_ms", 0), reverse=True)
                last_snap = existing_for_target[0]
                if last_snap.get("latest_mtime_ns") == latest_mtime_ns:
                    return {
                        **last_snap,
                        "status": "SKIPPED_UNCHANGED",
                    }

        timestamp_ms = int(time.time() * 1000)
        prefix = "MIM-AUTO" if is_auto else "MIM-SNAP"
        snapshot_id = f"{prefix}-{uuid.uuid4().hex[:8].upper()}"
        archive_path = self.snapshots_dir / f"{snapshot_id}.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if resolved_target.is_file():
                zf.write(resolved_target, arcname=resolved_target.name)
            else:
                for root, dirs, files in os.walk(resolved_target):
                    dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
                    for file in files:
                        if file.endswith((".pyc", ".db", ".db-wal", ".db-shm", ".webp")):
                            continue
                        full_file = Path(root) / file
                        rel_file = full_file.relative_to(resolved_target)
                        try:
                            zf.write(full_file, arcname=str(rel_file))
                        except Exception:
                            pass

        meta = {
            "status": "CREATED",
            "snapshot_id": snapshot_id,
            "mode": "USERSPACE_ARCHIVE",
            "is_auto": is_auto,
            "target_path": str(resolved_target),
            "archive_path": str(archive_path),
            "timestamp_ms": timestamp_ms,
            "latest_mtime_ns": latest_mtime_ns,
        }
        self._registry[snapshot_id] = meta
        self._save_registry()

        if is_auto:
            self._prune_auto_backups()

        return meta

    def list_snapshots(self) -> List[Dict[str, Any]]:
        self._registry = self._load_registry()
        items = list(self._registry.values())
        items.sort(key=lambda x: x.get("timestamp_ms", 0), reverse=True)
        return items

    def get_latest_snapshot_id(self) -> Optional[str]:
        snaps = self.list_snapshots()
        return snaps[0]["snapshot_id"] if snaps else None

    def rollback_snapshot(
        self,
        snapshot_id: Optional[str] = None,
        restore_target_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._registry = self._load_registry()
        target_sid = snapshot_id or self.get_latest_snapshot_id()
        if not target_sid:
            raise ValueError("No snapshots available to restore.")

        meta = self._registry.get(target_sid)
        archive_candidate = self.snapshots_dir / f"{target_sid}.zip"

        if (meta and Path(meta.get("archive_path", "")).exists()) or archive_candidate.exists():
            archive_path = Path(meta["archive_path"]) if meta else archive_candidate
            dest_str = restore_target_path or (meta.get("target_path") if meta else "")
            if not dest_str:
                raise ValueError("Restore target path is unknown for this snapshot.")

            dest = Path(dest_str).resolve()
            dest.mkdir(parents=True, exist_ok=True)

            for item in dest.iterdir():
                if item.name in self.IGNORE_DIRS or item.name.endswith((".db", ".db-wal", ".db-shm")):
                    continue
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception:
                    pass

            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest)

            return {
                "status": "ROLLED_BACK",
                "snapshot_id": target_sid,
                "restored_path": str(dest),
                "timestamp_ms": int(time.time() * 1000),
            }

        raise FileNotFoundError(f"Backup archive not found for {target_sid}")