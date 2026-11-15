import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyQt6.QtCore import Qt, QRectF, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QColor,
    QPixmap,
    QFont,
    QKeyEvent,
    QMouseEvent,
)
from PyQt6.QtWidgets import (
    QWidget,
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QLineEdit,
    QPushButton,
    QListWidget,
    QTextEdit,
    QDialog,
    QFileDialog,
    QApplication,
)

from mim.config import BASE_DIR
from mim.storage.sqlite_manager import SQLiteManager
from mim.search.hybrid_rag import HybridRAGEngine
from mim.system.vss_manager import VSSManager
from mim.ingestion.coordinator import IngestionCoordinator, SentinelState
from mim.ai.gemini_summarizer import GeminiSummarizer
from mim.ui.theme import TacticalTheme, apply_window_capture_cloaking


class AISummaryWorker(QThread):
    """Background worker thread that runs Gemini 2.5 Flash Q&A or Summary without freezing the UI."""
    finished_summary = pyqtSignal(dict)

    def __init__(
        self,
        sqlite_mgr: SQLiteManager,
        start_ts: int,
        end_ts: int,
        custom_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.sqlite_mgr = sqlite_mgr
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.custom_prompt = custom_prompt or (
            "Explain in simple, friendly English what I was doing on my computer, "
            "what apps/websites I used, and highlight any important text, tasks, or errors."
        )

    def run(self) -> None:
        try:
            summarizer = GeminiSummarizer(sqlite_mgr=self.sqlite_mgr)
            res = summarizer.summarize_window(
                start_ts=self.start_ts,
                end_ts=self.end_ts,
                prompt_focus=self.custom_prompt,
            )
            self.finished_summary.emit(res)
        except Exception as exc:
            self.finished_summary.emit({
                "engine": "error",
                "frame_count": 0,
                "summary": f"Could not generate answer: {exc}",
            })


class SummaryPopupDialog(QDialog):
    """
    Interactive AI History Chat & Summary Modal.
    Lets the user view the summary AND type any follow-up question across their entire history.
    """

    def __init__(
        self,
        parent: QWidget,
        sqlite_mgr: SQLiteManager,
        start_ts: int,
        end_ts: int,
        title_text: str,
        body_text: str,
    ):
        super().__init__(parent)
        self.sqlite_mgr = sqlite_mgr
        self.start_ts = start_ts
        self.end_ts = end_ts
        self._worker: Optional[AISummaryWorker] = None

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(680, 480)
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(10, 14, 20, 248);
                border: 2px solid #00E5FF;
                border-radius: 12px;
                color: #F3F3F3;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#ModalTitle {
                color: #FFB000;
                font-size: 15px;
                font-weight: bold;
            }
            QTextEdit {
                background-color: rgba(20, 25, 34, 235);
                color: #F3F3F3;
                border: 1px solid rgba(243, 243, 243, 50);
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
            }
            QLineEdit {
                background-color: rgba(20, 25, 34, 245);
                color: #F3F3F3;
                border: 1px solid #00E5FF;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QPushButton {
                background-color: rgba(0, 229, 255, 35);
                color: #00E5FF;
                border: 1px solid #00E5FF;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(0, 229, 255, 80);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        self.header = QLabel(title_text)
        self.header.setObjectName("ModalTitle")
        layout.addWidget(self.header)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setPlainText(body_text)
        layout.addWidget(self.text_box, stretch=1)

        ask_row = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText(
            "Ask anything about your history (e.g., 'What have I worked on the last 2 days?')..."
        )
        self.question_input.returnPressed.connect(self._on_ask_question)
        ask_row.addWidget(self.question_input, stretch=1)

        self.ask_btn = QPushButton("💬 Ask AI")
        self.ask_btn.clicked.connect(self._on_ask_question)
        ask_row.addWidget(self.ask_btn)
        layout.addLayout(ask_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        copy_btn = QPushButton("📋 Copy Text")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.text_box.toPlainText()))
        btn_row.addWidget(copy_btn)

        close_btn = QPushButton("✕ Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def _on_ask_question(self) -> None:
        question = self.question_input.text().strip()
        if not question:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self.ask_btn.setText("⏳ Asking...")
        self.question_input.setEnabled(False)
        self.text_box.append(f"\n\n🧑 You: {question}\n⏳ Gemini is scanning your multi-day & archival history...")

        # Scan up to 30 days of hourly rollups + all-time FTS5 search hits
        now_ms = int(time.time() * 1000)
        wide_start = min(self.start_ts, now_ms - (30 * 24 * 3600 * 1000))
        wide_end = max(self.end_ts, now_ms + 60_000)

        self._worker = AISummaryWorker(
            self.sqlite_mgr,
            wide_start,
            wide_end,
            custom_prompt=question,
        )
        self._worker.finished_summary.connect(self._on_answer_ready)
        self._worker.start()

    def _on_answer_ready(self, result: dict) -> None:
        self.ask_btn.setText("💬 Ask AI")
        self.question_input.setEnabled(True)
        self.question_input.clear()
        self.question_input.setFocus()

        engine = result.get("engine", "AI")
        count = result.get("frame_count", 0)
        answer = result.get("summary", "No answer returned.")
        self.header.setText(f"✨ AI History Assistant ({count} moments analyzed via {engine})")
        self.text_box.setPlainText(answer)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_window_capture_cloaking(int(self.winId()))


class ScrubberOverlayWindow(QWidget):
    """
    Full-screen Time-Scrubber Overlay with Interactive AI History Chat,
    Matrix Cyan text boxes, Cold Milestone Wireframe Reconstruction, and folder backup/restore.
    """

    def __init__(
        self,
        sqlite_mgr: SQLiteManager,
        rag_engine: Optional[HybridRAGEngine] = None,
        vss_mgr: Optional[VSSManager] = None,
        coordinator: Optional[IngestionCoordinator] = None,
    ):
        super().__init__(None)
        self.sqlite_mgr = sqlite_mgr
        self.rag_engine = rag_engine
        self.vss_mgr = vss_mgr or VSSManager()
        self.coordinator = coordinator

        self.frames: List[Dict[str, Any]] = []
        self.network_events: List[Dict[str, Any]] = []
        self.current_frame_idx: int = 0
        self.current_nodes: List[Dict[str, Any]] = []
        self.hovered_node: Optional[Dict[str, Any]] = None
        self.current_pixmap: Optional[QPixmap] = None
        self.toast_message: str = ""
        self.toast_expire_ms: int = 0
        self._ai_worker: Optional[AISummaryWorker] = None
        self._last_summary_range: tuple[int, int] = (0, 0)

        self._init_window()
        self._build_hud()

    def _init_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setStyleSheet(TacticalTheme.OVERLAY_STYLESHEET)

        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())
        else:
            self.setGeometry(0, 0, 1920, 1080)

    def _build_hud(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 28)

        top_row = QHBoxLayout()

        self.state_badge = QLabel("● Recording On")
        self.state_badge.setObjectName("StateBadgeActive")
        top_row.addWidget(self.state_badge)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("SearchInput")
        self.search_input.setPlaceholderText(
            "Search anything you saw on screen (even from months ago)... Press Enter"
        )
        self.search_input.setFixedWidth(420)
        self.search_input.returnPressed.connect(self._on_execute_search)
        top_row.addWidget(self.search_input)

        self.search_btn = QPushButton("🔍 Find on Screen")
        self.search_btn.setObjectName("ActionBtn")
        self.search_btn.clicked.connect(self._on_execute_search)
        top_row.addWidget(self.search_btn)

        top_row.addStretch(1)

        self.ai_summary_btn = QPushButton("✨ Ask AI / Summarize")
        self.ai_summary_btn.setObjectName("RollbackBtn")
        self.ai_summary_btn.setToolTip("Summarize your screen history or ask AI any question about what you did")
        self.ai_summary_btn.clicked.connect(self._on_request_ai_summary)
        top_row.addWidget(self.ai_summary_btn)

        self.toggle_log_btn = QPushButton("📋 Activity Log")
        self.toggle_log_btn.setObjectName("ActionBtn")
        self.toggle_log_btn.clicked.connect(self._toggle_telemetry_drawer)
        top_row.addWidget(self.toggle_log_btn)

        self.rollback_btn = QPushButton("💾 Backup Folder Here")
        self.rollback_btn.setObjectName("RollbackBtn")
        self.rollback_btn.clicked.connect(self._on_trigger_rollback)
        top_row.addWidget(self.rollback_btn)

        self.close_btn = QPushButton("✕ Exit [ESC]")
        self.close_btn.setObjectName("ActionBtn")
        self.close_btn.clicked.connect(self.hide)
        top_row.addWidget(self.close_btn)

        root_layout.addLayout(top_row)

        mid_row = QHBoxLayout()
        mid_row.addStretch(1)

        self.telemetry_panel = QFrame()
        self.telemetry_panel.setObjectName("TelemetryPanel")
        self.telemetry_panel.setFixedWidth(360)
        self.telemetry_panel.setMaximumHeight(420)
        tel_layout = QVBoxLayout(self.telemetry_panel)

        self.tel_header = QLabel("ACTIVITY & SEARCH RESULTS")
        self.tel_header.setObjectName("ProcessBadge")
        tel_layout.addWidget(self.tel_header)

        self.telemetry_list = QListWidget()
        self.telemetry_list.setObjectName("TelemetryList")
        self.telemetry_list.itemClicked.connect(self._on_telemetry_item_clicked)
        tel_layout.addWidget(self.telemetry_list)

        self.telemetry_panel.setVisible(False)

        mid_row.addWidget(self.telemetry_panel)
        root_layout.addLayout(mid_row)
        root_layout.addStretch(1)

        self.scrubber_bar = QWidget()
        self.scrubber_bar.setObjectName("ScrubberBar")
        self.scrubber_bar.setFixedHeight(74)
        bar_layout = QHBoxLayout(self.scrubber_bar)
        bar_layout.setContentsMargins(18, 10, 18, 10)
        bar_layout.setSpacing(16)

        self.process_label = QLabel("App: --")
        self.process_label.setObjectName("ProcessBadge")
        self.process_label.setFixedWidth(180)
        bar_layout.addWidget(self.process_label)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        bar_layout.addWidget(self.timeline_slider, stretch=1)

        self.timestamp_label = QLabel("--:--:-- | 0 text items")
        self.timestamp_label.setObjectName("MetaText")
        self.timestamp_label.setFixedWidth(250)
        bar_layout.addWidget(self.timestamp_label)

        root_layout.addWidget(self.scrubber_bar)

    def _toggle_telemetry_drawer(self) -> None:
        self.telemetry_panel.setVisible(not self.telemetry_panel.isVisible())

    def _on_request_ai_summary(self) -> None:
        if not self.frames:
            self.show_toast("No screen history recorded yet!")
            return
        if self._ai_worker is not None and self._ai_worker.isRunning():
            self.show_toast("AI is already thinking... please wait!")
            return

        center_ts = self.frames[self.current_frame_idx]["timestamp_ms"]
        start_ts = center_ts - (60 * 60 * 1000)
        end_ts = center_ts + (15 * 60 * 1000)
        self._last_summary_range = (start_ts, end_ts)

        self.ai_summary_btn.setText("⏳ Thinking...")
        self.show_toast("Asking Gemini AI to analyze your screen history...")

        self._ai_worker = AISummaryWorker(self.sqlite_mgr, start_ts, end_ts)
        self._ai_worker.finished_summary.connect(self._on_ai_summary_ready)
        self._ai_worker.start()

    def _on_ai_summary_ready(self, result: dict) -> None:
        self.ai_summary_btn.setText("✨ Ask AI / Summarize")
        engine = result.get("engine", "AI")
        count = result.get("frame_count", 0)
        summary_text = result.get("summary", "No summary returned.")
        start_ts, end_ts = self._last_summary_range

        dialog = SummaryPopupDialog(
            self,
            sqlite_mgr=self.sqlite_mgr,
            start_ts=start_ts,
            end_ts=end_ts,
            title_text=f"✨ AI History Assistant ({count} moments analyzed via {engine})",
            body_text=summary_text,
        )
        dialog.exec()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_window_capture_cloaking(int(self.winId()))
        self.refresh_timeline_data()

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.showFullScreen()
            self.raise_()
            self.activateWindow()

    def refresh_timeline_data(self) -> None:
        conn = self.sqlite_mgr._get_conn()
        # Load the newest 1000 frames in chronological order for smooth scrubbing
        rows = conn.execute("""
            SELECT * FROM (
                SELECT * FROM frames
                WHERE NOT (LOWER(process_name) = 'python.exe' AND LOWER(window_title) = 'python')
                ORDER BY timestamp_ms DESC
                LIMIT 1000
            ) ORDER BY timestamp_ms ASC
        """).fetchall()
        if not rows:
            rows = conn.execute("SELECT * FROM frames ORDER BY timestamp_ms ASC LIMIT 500").fetchall()
        self.frames = [dict(r) for r in rows]

        net_rows = conn.execute(
            "SELECT * FROM network_events ORDER BY timestamp_ms DESC LIMIT 50"
        ).fetchall()
        self.network_events = [dict(r) for r in net_rows]

        is_rec = True
        if self.coordinator is not None:
            is_rec = self.coordinator.state == SentinelState.ACTIVE
        if is_rec:
            self.state_badge.setText("● Recording On")
            self.state_badge.setObjectName("StateBadgeActive")
        else:
            self.state_badge.setText("⏸ Recording Paused")
            self.state_badge.setObjectName("StateBadgePaused")
        self.state_badge.style().unpolish(self.state_badge)
        self.state_badge.style().polish(self.state_badge)

        if not self.frames:
            self.timeline_slider.setMaximum(0)
            self.current_nodes = []
            self.current_pixmap = None
            self.process_label.setText("No history yet")
            self.update()
            return

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(len(self.frames) - 1)
        self.timeline_slider.setValue(len(self.frames) - 1)
        self.timeline_slider.blockSignals(False)

        self.load_frame_at_index(len(self.frames) - 1)

    def _jump_to_or_inject_frame(self, frame_id: str) -> bool:
        """
        Jumps to `frame_id` if already in `self.frames`, or dynamically loads an older archival
        frame from months ago into the timeline so you can view its Milestone photo or Wireframe!
        """
        for idx, f in enumerate(self.frames):
            if f["frame_id"] == frame_id or f["frame_id"].startswith(frame_id):
                self.timeline_slider.setValue(idx)
                return True

        conn = self.sqlite_mgr._get_conn()
        row = conn.execute(
            "SELECT * FROM frames WHERE frame_id LIKE ? LIMIT 1",
            (f"{frame_id}%",),
        ).fetchone()
        if not row:
            return False

        target_frame = dict(row)
        self.frames.insert(0, target_frame)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMaximum(len(self.frames) - 1)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.load_frame_at_index(0)
        return True

    def load_frame_at_index(self, index: int) -> None:
        if not self.frames or index < 0 or index >= len(self.frames):
            return

        self.current_frame_idx = index
        frame = self.frames[index]
        fid = frame["frame_id"]

        img_path = BASE_DIR / frame["image_path"]
        if img_path.exists():
            self.current_pixmap = QPixmap(str(img_path))
        else:
            self.current_pixmap = None

        self.current_nodes = self.sqlite_mgr.get_nodes_for_frame(fid)
        self.hovered_node = None

        dt_str = (
            datetime.fromtimestamp(frame["timestamp_ms"] / 1000.0)
            .strftime("%b %d, %I:%M:%S %p")
        )
        clean_app = frame["process_name"].replace(".exe", "").capitalize()
        self.process_label.setText(f"App: {clean_app[:16]}")

        snap_id = frame.get("vss_snapshot_id")
        if snap_id:
            self.rollback_btn.setText(f"⏪ Restore Files ({snap_id[-4:]})")
            snap_tag = " | 💾 Backup"
        else:
            self.rollback_btn.setText("💾 Backup Folder Here")
            snap_tag = ""

        text_box_count = sum(
            1 for n in self.current_nodes if (n.get("content_text") or "").strip()
        )
        self.timestamp_label.setText(
            f"{dt_str} | {text_box_count} items{snap_tag}"
        )

        self.telemetry_list.clear()
        for node in self.current_nodes[:15]:
            txt = (node.get("content_text") or "").strip()
            if txt:
                self.telemetry_list.addItem(f"📝 Text: {txt[:60]}")

        f_ts = frame["timestamp_ms"]
        for ev in self.network_events:
            if abs(ev["timestamp_ms"] - f_ts) <= 30_000:
                app_clean = ev["process_name"].replace(".exe", "")
                self.telemetry_list.addItem(
                    f"🌐 {app_clean} connected to {ev['dest_ip']}"
                )

        self.update()

    def _on_slider_changed(self, value: int) -> None:
        self.load_frame_at_index(value)

    def _on_execute_search(self) -> None:
        query = self.search_input.text().strip()
        if not query:
            return

        self.telemetry_list.clear()
        if self.rag_engine is not None:
            hits = self.rag_engine.search(query=query, limit=15)
        else:
            hits = self.sqlite_mgr.search_lexical_bm25(query=query, limit=15)

        if not hits:
            self.show_toast(f"Nothing found for '{query}'")
            return

        self.telemetry_panel.setVisible(True)
        self.tel_header.setText(f"SEARCH RESULTS ({len(hits)})")

        for h in hits:
            fid = h["frame_id"]
            app_clean = h["process_name"].replace(".exe", "").capitalize()
            snippet = (
                h.get("matched_text")
                or h.get("content_text")
                or h.get("window_title")
                or ""
            )[:50]
            self.telemetry_list.addItem(f"🔍 [{fid[:8]}] {app_clean}: {snippet}")

        top_fid = hits[0]["frame_id"]
        self._jump_to_or_inject_frame(top_fid)
        self.show_toast(f"Found {len(hits)} matches! Jumped to best match.")

    def _on_telemetry_item_clicked(self, item) -> None:
        text = item.text()
        if "[" in text and "]" in text:
            short_id = text.split("[")[1].split("]")[0]
            self._jump_to_or_inject_frame(short_id)
        else:
            clean_copy = text.replace("📝 Text: ", "")
            QApplication.clipboard().setText(clean_copy)
            self.show_toast("Copied to clipboard!")

    def _on_trigger_rollback(self) -> None:
        if not self.frames:
            return
        frame = self.frames[self.current_frame_idx]
        snap_id = frame.get("vss_snapshot_id")

        if not snap_id:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Choose a Folder to Backup at This Moment",
                str(BASE_DIR),
            )
            if not folder:
                return

            meta = self.vss_mgr.create_snapshot(target_path=folder)
            snap_id = meta["snapshot_id"]
            frame["vss_snapshot_id"] = snap_id

            conn = self.sqlite_mgr._get_conn()
            with self.sqlite_mgr._write_lock:
                conn.execute(
                    "UPDATE frames SET vss_snapshot_id = ? WHERE frame_id = ?",
                    (snap_id, frame["frame_id"]),
                )
                conn.commit()

            self.rollback_btn.setText(f"⏪ Restore Files ({snap_id[-4:]})")
            self.show_toast(
                f"Backed up '{Path(folder).name}'! Click again anytime to restore."
            )
            return

        try:
            res = self.vss_mgr.rollback_snapshot(snapshot_id=snap_id)
            folder_name = Path(res["restored_path"]).name
            self.show_toast(f"Restored '{folder_name}' back to this backup point!")
        except Exception as exc:
            self.show_toast(f"Restore error: {exc}")

    def show_toast(self, message: str, duration_ms: int = 2600) -> None:
        self.toast_message = message
        self.toast_expire_ms = int(time.time() * 1000) + duration_ms
        self.update()
        QTimer.singleShot(duration_ms + 50, self.update)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        mx, my = int(pos.x()), int(pos.y())

        best_node: Optional[Dict[str, Any]] = None
        best_area = float("inf")

        for node in self.current_nodes:
            bbox = node.get("bounding_box", [0, 0, 0, 0])
            if len(bbox) == 4:
                left, top, right, bottom = bbox
                if left <= mx <= right and top <= my <= bottom:
                    area = max(1, (right - left) * (bottom - top))
                    if node.get("content_text"):
                        area *= 0.25
                    if area < best_area:
                        best_area = area
                        best_node = node

        if best_node != self.hovered_node:
            self.hovered_node = best_node
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.hovered_node:
            text_to_copy = (self.hovered_node.get("content_text") or "").strip()
            if text_to_copy:
                QApplication.clipboard().setText(text_to_copy)
                self.show_toast(f'Copied: "{text_to_copy[:40]}..."')

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        elif event.key() == Qt.Key.Key_Left:
            self.timeline_slider.setValue(max(0, self.current_frame_idx - 1))
        elif event.key() == Qt.Key.Key_Right:
            self.timeline_slider.setValue(
                min(len(self.frames) - 1, self.current_frame_idx + 1)
            )
        elif event.key() == Qt.Key.Key_C and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            if self.hovered_node and self.hovered_node.get("content_text"):
                txt = self.hovered_node["content_text"]
                QApplication.clipboard().setText(txt)
                self.show_toast(f'Copied: "{txt[:40]}..."')

    def _draw_tactical_wireframe_background(self, painter: QPainter) -> None:
        """
        Renders a Tactical Blueprint Wireframe reconstruction from the saved SQLite `ui_nodes`
        when viewing an older Cold Tier intermediate frame whose `.webp` was pruned to save disk space.
        """
        painter.fillRect(self.rect(), QColor(8, 12, 18, 245))

        # Subtle blueprint grid lines
        grid_pen = QPen(QColor(0, 229, 255, 18), 1.0)
        painter.setPen(grid_pen)
        w, h = self.width(), self.height()
        for x in range(0, w, 64):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, 64):
            painter.drawLine(0, y, w, y)

        if self.frames and 0 <= self.current_frame_idx < len(self.frames):
            f = self.frames[self.current_frame_idx]
            header_txt = (
                f"📐 ARCHIVAL BLUEPRINT RECONSTRUCTION — {f.get('process_name', '')} | "
                f"{(f.get('window_title') or '')[:70]}"
            )
            painter.setPen(TacticalTheme.TERMINAL_AMBER)
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.drawText(QRectF(28, 78, w - 56, 28), Qt.AlignmentFlag.AlignLeft, header_txt)

        # Render UIA structural containers and text labels inside their original screen coordinates
        box_pen = QPen(QColor(0, 229, 255, 95), 1.0)
        box_fill = QBrush(QColor(16, 26, 38, 175))
        painter.setFont(QFont("Segoe UI", 9))

        for node in self.current_nodes:
            bbox = node.get("bounding_box", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            l, t, r, b = bbox
            bw, bh = r - l, b - t
            if bw <= 4 or bh <= 4:
                continue

            rect_f = QRectF(l, t, bw, bh)
            painter.setPen(box_pen)
            painter.setBrush(box_fill)
            painter.drawRoundedRect(rect_f, 3.0, 3.0)

            txt = (node.get("content_text") or "").strip()
            if txt and bw > 24 and bh > 12:
                painter.setPen(QColor(225, 238, 248, 225))
                painter.drawText(
                    rect_f.adjusted(5.0, 2.0, -5.0, -2.0),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    txt[:90],
                )

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        if self.current_pixmap is not None and not self.current_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.current_pixmap)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 85))

            subtle_pen = QPen(
                TacticalTheme.MATRIX_CYAN_SUBTLE, 1.0, Qt.PenStyle.DashLine
            )
            painter.setPen(subtle_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for node in self.current_nodes:
                if not node.get("content_text"):
                    continue
                bbox = node.get("bounding_box", [0, 0, 0, 0])
                if len(bbox) == 4:
                    l, t, r, b = bbox
                    if r > l and b > t:
                        painter.drawRect(QRectF(l, t, r - l, b - t))
        else:
            # Cold Tier intermediate frame: render the Tactical Blueprint Wireframe!
            self._draw_tactical_wireframe_background(painter)

        if self.hovered_node is not None:
            bbox = self.hovered_node.get("bounding_box", [0, 0, 0, 0])
            if len(bbox) == 4:
                l, t, r, b = bbox
                rect_f = QRectF(l, t, max(10, r - l), max(10, b - t))
                active_pen = QPen(TacticalTheme.MATRIX_CYAN, 2.0)
                painter.setPen(active_pen)
                painter.setBrush(QBrush(TacticalTheme.MATRIX_CYAN_FILL))
                painter.drawRect(rect_f)

                txt = (self.hovered_node.get("content_text") or "Item").strip()
                banner_rect = QRectF(
                    l,
                    max(8.0, t - 28.0),
                    min(520.0, max(180.0, len(txt) * 7.5)),
                    24.0,
                )
                painter.setPen(QPen(TacticalTheme.MATRIX_CYAN, 1.0))
                painter.setBrush(QBrush(TacticalTheme.PITCH_BLACK_88))
                painter.drawRoundedRect(banner_rect, 4.0, 4.0)

                painter.setPen(TacticalTheme.GHOST_WHITE)
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(
                    banner_rect.adjusted(8.0, 0.0, -8.0, 0.0),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    f"Click to Copy: {txt[:60]}",
                )

        now_ms = int(time.time() * 1000)
        if self.toast_message and now_ms < self.toast_expire_ms:
            tw = 480.0
            th = 36.0
            tx = (self.width() - tw) / 2.0
            ty = 82.0
            toast_rect = QRectF(tx, ty, tw, th)
            painter.setPen(QPen(TacticalTheme.TERMINAL_AMBER, 1.5))
            painter.setBrush(QBrush(TacticalTheme.PITCH_BLACK_88))
            painter.drawRoundedRect(toast_rect, 6.0, 6.0)

            painter.setPen(TacticalTheme.TERMINAL_AMBER)
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.drawText(
                toast_rect, Qt.AlignmentFlag.AlignCenter, self.toast_message
            )