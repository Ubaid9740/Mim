import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Set

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
SQLITE_DB_PATH = DATA_DIR / "mim_temporal.db"
LANCEDB_URI = DATA_DIR / "mim_lancedb"


@dataclass(frozen=True)
class MimConfig:
    """Centralized configuration for the Mim developer observability & MCP substrate."""

    data_dir: Path = DATA_DIR
    frames_dir: Path = FRAMES_DIR
    sqlite_path: Path = SQLITE_DB_PATH
    lancedb_uri: Path = LANCEDB_URI

    # Capture & Native Quality Parameters
    capture_fps: float = 2.0
    dhash_threshold: int = 6
    uia_max_depth: int = 5
    uia_scrape_interval_sec: float = 2.5
    jpeg_quality: int = 85                # Crisp 100% native resolution for Hot Tier
    warm_tier_quality: int = 50           # Archival WebP quality for older frames
    embedding_dim: int = 384

    # Tiered Storage, Batching & Dirty-Aware Auto-Backup Retention
    hot_tier_frames: int = 150            # Keep newest 150 frames at 100% quality=85
    max_stored_frames: int = 5000         # Stores ~2-4 days of active delta frames (~175 MB)
    lancedb_batch_size: int = 20          # Buffer vectors in RAM before flushing to disk
    auto_backup_interval_sec: int = 600   # Check workspace for file modifications every 10 min
    max_auto_backups: int = 5             # Keep max 5 auto-backup ZIPs on disk

    # Zero-Trust Process Blacklist
    blacklisted_processes: Set[str] = field(default_factory=lambda: {
        "bitwarden.exe",
        "1password.exe",
        "keepass.exe",
        "keepassxc.exe",
        "dashlane.exe",
        "lastpass.exe",
        "credentialuibroker.exe",
        "whatsapp.exe",
        "telegram.exe",
    })

    # Zero-Trust Window Title Keyword Blacklist
    blacklisted_title_keywords: Set[str] = field(default_factory=lambda: {
        "incognito",
        "inprivate",
        "private browsing",
        "password manager",
        "bank",
        "checkout",
        ".env",
        "whatsapp",
        "telegram",
        "gmail",
        "discord",
    })

    api_host: str = "127.0.0.1"
    api_port: int = 8420
    scrubber_hotkey: str = "<ctrl>+<shift>+<space>"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_uri.mkdir(parents=True, exist_ok=True)


CONFIG = MimConfig()
CONFIG.ensure_directories()