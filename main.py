import sys
import ctypes
import signal
import queue
import threading
from typing import Optional
import uvicorn
from pynput import keyboard

from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QBrush, QPen, QAction
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu


def enable_win32_per_monitor_dpi() -> None:
    """
    Forces Per-Monitor V2 DPI Awareness across Win32, MSS, UIAutomation, and PyQt6
    so Matrix Cyan UIA bounding boxes align 1:1 with physical pixels on 125%/150% displays.
    Must execute before QApplication is instantiated.
    """
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


# Lock DPI awareness before importing/initializing GUI components
enable_win32_per_monitor_dpi()

from mim.config import CONFIG
from mim.storage.sqlite_manager import SQLiteManager
from mim.storage.lancedb_manager import LanceDBManager
from mim.search.hybrid_rag import HybridRAGEngine
from mim.system.vss_manager import VSSManager
from mim.system.net_tracer import NetworkTracer
from mim.ingestion.coordinator import IngestionCoordinator, SentinelState
from mim.api.server import create_mim_app
from mim.ui.theme import TacticalTheme
from mim.ui.eye_sentinel import EyeSentinelWidget
from mim.ui.scrubber_overlay import ScrubberOverlayWindow


class HotkeyBridge(QObject):
    """Thread-safe Qt signal bridge so pynput's background thread can trigger PyQt6 UI actions."""
    toggle_scrubber_signal = pyqtSignal()
    toggle_sentinel_signal = pyqtSignal()


class UvicornDaemon:
    """Runs the FastAPI Agent Bridge in a clean background daemon thread with graceful shutdown."""

    def __init__(self, app, host: str = CONFIG.api_host, port: int = CONFIG.api_port):
        self.config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(self.config)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self.server.run,
            name="MimFastAPIWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self.server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)


class MimOrchestrator:
    """
    Master runtime controller for Mim (Mimir).
    Single-owner process for SQLite WAL & LanceDB with an asynchronous embedding queue.
    """

    def __init__(self, start_api: bool = True, enable_hotkeys: bool = True):
        CONFIG.ensure_directories()

        # 1. Initialize Storage & Analytical Cores
        self.sqlite_mgr = SQLiteManager()
        self.lancedb_mgr = LanceDBManager()
        self.rag_engine = HybridRAGEngine(
            sqlite_mgr=self.sqlite_mgr,
            lancedb_mgr=self.lancedb_mgr,
        )
        self.vss_mgr = VSSManager()
        self.net_tracer = NetworkTracer(sqlite_mgr=self.sqlite_mgr, poll_interval=3.0)

        # 2. Asynchronous Embedding Queue & Background Consumer Thread
        self._embed_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=250)
        self._embed_stop = threading.Event()
        self._embed_thread: Optional[threading.Thread] = None

        # 3. Initialize Ingestion Coordinator
        self.coordinator = IngestionCoordinator(
            sqlite_mgr=self.sqlite_mgr,
            lancedb_mgr=self.lancedb_mgr,
            vss_mgr=self.vss_mgr,
            on_frame_committed=self._enqueue_frame_for_embedding,
        )

        # 4. Initialize FastAPI Agent Bridge
        self.fastapi_app = create_mim_app(
            sqlite_mgr=self.sqlite_mgr,
            lancedb_mgr=self.lancedb_mgr,
            coordinator=self.coordinator,
            rag_engine=self.rag_engine,
            vss_mgr=self.vss_mgr,
            net_tracer=self.net_tracer,
        )
        self.api_daemon = UvicornDaemon(self.fastapi_app) if start_api else None

        # 5. Initialize PyQt6 Tactical Observability Widgets
        self.hotkey_bridge = HotkeyBridge()
        self.eye_sentinel = EyeSentinelWidget(coordinator=self.coordinator)
        self.scrubber_overlay = ScrubberOverlayWindow(
            sqlite_mgr=self.sqlite_mgr,
            rag_engine=self.rag_engine,
            vss_mgr=self.vss_mgr,
            coordinator=self.coordinator,
        )

        self.eye_sentinel.request_open_scrubber.connect(self.scrubber_overlay.toggle_visibility)
        self.eye_sentinel.state_changed.connect(self._on_sentinel_state_changed)
        self.hotkey_bridge.toggle_scrubber_signal.connect(self.scrubber_overlay.toggle_visibility)
        self.hotkey_bridge.toggle_sentinel_signal.connect(self.eye_sentinel.toggle_sentinel)

        # 6. System Tray & Global Hotkeys
        self.tray_icon: Optional[QSystemTrayIcon] = None
        self.hotkey_listener: Optional[keyboard.GlobalHotKeys] = None
        self._enable_hotkeys = enable_hotkeys
        self._is_shutting_down = False

        self._setup_system_tray()
        if self._enable_hotkeys:
            self._setup_global_hotkeys()

    def _enqueue_frame_for_embedding(self, frame_record: dict, node_count: int) -> None:
        """Non-blocking enqueue so the 2-FPS capture thread never waits on ONNX inference."""
        try:
            self._embed_queue.put_nowait(frame_record)
        except queue.Full:
            pass

    def _embedding_worker_loop(self) -> None:
        """Background consumer thread that embeds UIA text into LanceDB asynchronously."""
        while not self._embed_stop.is_set():
            try:
                frame_record = self._embed_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame_record is None:
                break

            try:
                nodes = self.sqlite_mgr.get_nodes_for_frame(frame_record["frame_id"])
                self.rag_engine.index_frame_semantics(frame_record, nodes)
            except Exception as exc:
                print(f"[MimEmbedWorker] Warning: {exc}")
            finally:
                self._embed_queue.task_done()

    @staticmethod
    def _create_tray_icon(active: bool = True) -> QIcon:
        pix = QPixmap(32, 32)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        ring_color = TacticalTheme.TERMINAL_AMBER if active else TacticalTheme.GHOST_WHITE
        painter.setPen(QPen(ring_color, 2.5))
        painter.setBrush(QBrush(TacticalTheme.PITCH_BLACK_88))
        painter.drawEllipse(2, 2, 28, 28)

        if active:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(TacticalTheme.EYE_IRIS_OUTER))
            painter.drawEllipse(9, 9, 14, 14)
            painter.setBrush(QBrush(TacticalTheme.EYE_PUPIL))
            painter.drawEllipse(13, 13, 6, 6)
        else:
            painter.setPen(QPen(TacticalTheme.GHOST_WHITE, 2.5))
            painter.drawLine(8, 16, 24, 16)

        painter.end()
        return QIcon(pix)

    def _setup_system_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self.tray_icon = QSystemTrayIcon(self._create_tray_icon(active=True))
        self.tray_icon.setToolTip("Mim (Mimir) — Tactical AI Flight Recorder [ACTIVE]")

        menu = QMenu()
        self.action_toggle_scrubber = QAction("Open Time-Scrubber (Ctrl+Shift+Space)", menu)
        self.action_toggle_scrubber.triggered.connect(self.scrubber_overlay.toggle_visibility)
        menu.addAction(self.action_toggle_scrubber)

        self.action_toggle_sentinel = QAction("Pause / Resume Recording (Ctrl+Shift+P)", menu)
        self.action_toggle_sentinel.triggered.connect(self.eye_sentinel.toggle_sentinel)
        menu.addAction(self.action_toggle_sentinel)

        self.action_snapshot = QAction("Create Workspace Backup Checkpoint", menu)
        self.action_snapshot.triggered.connect(self._trigger_manual_snapshot)
        menu.addAction(self.action_snapshot)

        menu.addSeparator()
        self.action_exit = QAction("Shutdown Mim", menu)
        self.action_exit.triggered.connect(self.shutdown)
        menu.addAction(self.action_exit)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.show()

    def _setup_global_hotkeys(self) -> None:
        try:
            self.hotkey_listener = keyboard.GlobalHotKeys({
                CONFIG.scrubber_hotkey: self.hotkey_bridge.toggle_scrubber_signal.emit,
                "<ctrl>+<shift>+p": self.hotkey_bridge.toggle_sentinel_signal.emit,
            })
            self.hotkey_listener.daemon = True
            self.hotkey_listener.start()
        except Exception as exc:
            print(f"[MimOrchestrator] Global hotkey registration warning: {exc}")

    def _on_sentinel_state_changed(self, new_state: str) -> None:
        is_active = (new_state == SentinelState.ACTIVE.value)
        if self.tray_icon is not None:
            self.tray_icon.setIcon(self._create_tray_icon(active=is_active))
            self.tray_icon.setToolTip(f"Mim (Mimir) — Flight Recorder [{new_state}]")

    def _trigger_manual_snapshot(self) -> None:
        meta = self.vss_mgr.create_snapshot()
        if self.tray_icon is not None:
            self.tray_icon.showMessage(
                "Mim Backup Created",
                f"Snapshot ID: {meta['snapshot_id']}",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )

    def start(self) -> None:
        print("=================================================================")
        print("  MIM (MIMIR) — DEVELOPER BLACK BOX & MCP MEMORY SUBSTRATE V1.0")
        print("=================================================================")
        print(f"  [+] SQLite WAL Database : {CONFIG.sqlite_path}")
        print(f"  [+] LanceDB Vector Store: {CONFIG.lancedb_uri}")
        print(f"  [+] Local Agent Bridge  : http://{CONFIG.api_host}:{CONFIG.api_port}")
        print(f"  [+] Time-Scrubber Hotkey: {CONFIG.scrubber_hotkey}")
        print("  [+] Privacy Toggle      : Left-Click Eye Sentinel or <ctrl>+<shift>+p")
        print("=================================================================")

        self._embed_stop.clear()
        self._embed_thread = threading.Thread(
            target=self._embedding_worker_loop,
            name="MimAsyncEmbedWorker",
            daemon=True,
        )
        self._embed_thread.start()

        if self.api_daemon is not None:
            self.api_daemon.start()
        self.net_tracer.start()
        self.coordinator.start()
        self.eye_sentinel.show()

    def shutdown(self) -> None:
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        print("\n[MimOrchestrator] Initiating graceful shutdown...")

        if self.hotkey_listener is not None:
            try:
                self.hotkey_listener.stop()
            except Exception:
                pass

        self.coordinator.stop()
        self.net_tracer.stop()

        # Stop async embedding worker and flush LanceDB batch buffer
        self._embed_stop.set()
        try:
            self._embed_queue.put_nowait(None)
        except Exception:
            pass
        if self._embed_thread and self._embed_thread.is_alive():
            self._embed_thread.join(timeout=3.0)
        try:
            self.lancedb_mgr.flush()
        except Exception:
            pass

        if self.api_daemon is not None:
            self.api_daemon.stop()

        self.scrubber_overlay.close()
        self.eye_sentinel.close()
        if self.tray_icon is not None:
            self.tray_icon.hide()

        self.sqlite_mgr.close()
        print("[MimOrchestrator] SQLite WAL & LanceDB buffers flushed cleanly.")

        app = QApplication.instance()
        if app is not None:
            app.quit()


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    orchestrator = MimOrchestrator(start_api=True, enable_hotkeys=True)

    def sigint_handler(*_args):
        orchestrator.shutdown()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    orchestrator.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()