from pathlib import Path
from typing import Optional, Callable
from PyQt6.QtCore import Qt, QPoint, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPainter,
    QPen,
    QBrush,
    QRadialGradient,
    QPainterPath,
    QPixmap,
    QMouseEvent,
)
from PyQt6.QtWidgets import QWidget, QApplication

from mim.ingestion.coordinator import IngestionCoordinator, SentinelState
from mim.ui.theme import TacticalTheme, apply_window_capture_cloaking


class EyeSentinelWidget(QWidget):
    """
    Persistent, frameless, hardware-cloaked privacy toggle widget.
    - Open State (ACTIVE): Blue concentric iris, dark pupil, rectangular top-right glint,
      and pulsing Terminal Amber (#FFB000) status ring.
    - Closed State (PAUSED): Soft gray sclera with curved sleeping eyelid & eyelashes,
      and Ghost White (#F3F3F3) safe ring.
    - Left-click toggles ACTIVE <-> PAUSED_MANUAL.
    - Drag with mouse to reposition anywhere on desktop; Right-click opens Time-Scrubber.
    """

    state_changed = pyqtSignal(str)
    request_open_scrubber = pyqtSignal()

    def __init__(
        self,
        coordinator: Optional[IngestionCoordinator] = None,
        open_icon_path: Optional[Path] = None,
        closed_icon_path: Optional[Path] = None,
        size: int = 68,
    ):
        super().__init__(None)
        self.coordinator = coordinator
        self.widget_size = size
        self._is_active = True
        self._pulse_phase = 0.0
        self._drag_start_pos: Optional[QPoint] = None
        self._did_drag = False

        # Optional custom image assets if placed in data/assets/
        self._open_pixmap = (
            QPixmap(str(open_icon_path))
            if open_icon_path and open_icon_path.exists()
            else None
        )
        self._closed_pixmap = (
            QPixmap(str(closed_icon_path))
            if closed_icon_path and closed_icon_path.exists()
            else None
        )

        self._configure_window()

        # Subtle pulse timer for active state synchronization
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(250)

    def _configure_window(self) -> None:
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(self.widget_size, self.widget_size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "Mim Eye Sentinel\n"
            "• Left-Click: Toggle Recording (Active / Paused)\n"
            "• Right-Click: Open Time-Scrubber Overlay (Ctrl+Shift+Space)\n"
            "• Drag: Move Sentinel"
        )

        # Position in top-right corner of primary screen by default
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.right() - self.widget_size - 24, geo.top() + 24)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_window_capture_cloaking(int(self.winId()))

    @property
    def is_recording_active(self) -> bool:
        if self.coordinator is not None:
            return self.coordinator.state == SentinelState.ACTIVE
        return self._is_active

    def _on_tick(self) -> None:
        if self.coordinator is not None:
            current = self.coordinator.state == SentinelState.ACTIVE
            if current != self._is_active:
                self._is_active = current
                self.state_changed.emit(self.coordinator.state.value)
        self._pulse_phase = (self._pulse_phase + 0.25) % 1.0
        self.update()

    def toggle_sentinel(self) -> str:
        """Toggles the Eye Sentinel between ACTIVE and PAUSED_MANUAL."""
        if self.coordinator is not None:
            res = self.coordinator.toggle_manual_pause(bind_to_active_pid=True)
            self._is_active = res["state"] == SentinelState.ACTIVE.value
            new_state = res["state"]
        else:
            self._is_active = not self._is_active
            new_state = "ACTIVE" if self._is_active else "PAUSED_MANUAL"

        self.state_changed.emit(new_state)
        self.update()
        return new_state

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._did_drag = False
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.request_open_scrubber.emit()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = (event.globalPosition().toPoint() - self._drag_start_pos) - self.pos()
            if delta.manhattanLength() > 4:
                self._did_drag = True
                self.move(event.globalPosition().toPoint() - self._drag_start_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._did_drag:
                self.toggle_sentinel()
            self._drag_start_pos = None
            self._did_drag = False
            event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = float(self.width())
        h = float(self.height())
        center = QRectF(6.0, 6.0, w - 12.0, h - 12.0)

        # 1. Draw Tactical Outer Status Ring (#FFB000 Terminal Amber vs #F3F3F3 Ghost White)
        if self.is_recording_active:
            ring_pen = QPen(TacticalTheme.TERMINAL_AMBER, 2.5)
        else:
            ring_pen = QPen(TacticalTheme.GHOST_WHITE_DIM, 2.0)
        painter.setPen(ring_pen)
        painter.setBrush(QBrush(TacticalTheme.PITCH_BLACK_70))
        painter.drawEllipse(QRectF(2.0, 2.0, w - 4.0, h - 4.0))

        # 2. Draw Sclera Sphere (Radial Gradient matching user reference image)
        grad = QRadialGradient(w / 2.0, h / 2.0, (w - 12.0) / 2.0)
        grad.setColorAt(0.0, TacticalTheme.EYE_SCLERA_INNER)
        grad.setColorAt(0.75, TacticalTheme.EYE_SCLERA_INNER)
        grad.setColorAt(1.0, TacticalTheme.EYE_SCLERA_OUTER)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(center)

        if self.is_recording_active:
            if self._open_pixmap is not None:
                painter.drawPixmap(center.toRect(), self._open_pixmap)
            else:
                self._paint_open_eye(painter, w, h)
        else:
            if self._closed_pixmap is not None:
                painter.drawPixmap(center.toRect(), self._closed_pixmap)
            else:
                self._paint_closed_eye(painter, w, h)

        # 3. Draw Top Status LED Dot
        led_color = TacticalTheme.TERMINAL_AMBER if self.is_recording_active else TacticalTheme.GHOST_WHITE
        painter.setBrush(QBrush(led_color))
        painter.drawEllipse(QRectF(w / 2.0 - 3.0, 1.0, 6.0, 6.0))

    def _paint_open_eye(self, painter: QPainter, w: float, h: float) -> None:
        """Vector recreation of the Open Eye Sentinel design."""
        cx, cy = w / 2.0, h / 2.0

        # Outer Blue Iris Ring
        r_outer_iris = (w - 12.0) * 0.33
        painter.setBrush(QBrush(TacticalTheme.EYE_IRIS_OUTER))
        painter.drawEllipse(QRectF(cx - r_outer_iris, cy - r_outer_iris, r_outer_iris * 2, r_outer_iris * 2))

        # Inner Darker Blue Iris Ring
        r_inner_iris = (w - 12.0) * 0.27
        painter.setBrush(QBrush(TacticalTheme.EYE_IRIS_INNER))
        painter.drawEllipse(QRectF(cx - r_inner_iris, cy - r_inner_iris, r_inner_iris * 2, r_inner_iris * 2))

        # Deep Navy/Black Pupil
        r_pupil = (w - 12.0) * 0.13
        painter.setBrush(QBrush(TacticalTheme.EYE_PUPIL))
        painter.drawEllipse(QRectF(cx - r_pupil, cy - r_pupil, r_pupil * 2, r_pupil * 2))

        # Rectangular Top-Right Glint (exact match to user's eye reference image)
        glint_w = (w - 12.0) * 0.15
        glint_h = (h - 12.0) * 0.23
        painter.setBrush(QBrush(TacticalTheme.EYE_GLINT))
        painter.drawRect(QRectF(cx + r_pupil * 0.95, cy - r_outer_iris * 1.05, glint_w, glint_h))

    def _paint_closed_eye(self, painter: QPainter, w: float, h: float) -> None:
        """Vector recreation of the Closed Eye Sentinel design (curved eyelid + 4 lashes)."""
        cx, cy = w / 2.0, h / 2.0

        # Subtle inner shading circle
        r_shade = (w - 12.0) * 0.30
        painter.setBrush(QBrush(TacticalTheme.EYE_SCLERA_OUTER))
        painter.drawEllipse(QRectF(cx - r_shade, cy - r_shade, r_shade * 2, r_shade * 2))

        # Curved Sleeping Eyelid Path
        pen = QPen(TacticalTheme.EYE_CLOSED_LINE, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        span_x = (w - 12.0) * 0.30
        lid_path = QPainterPath()
        lid_path.moveTo(cx - span_x, cy - 1.0)
        lid_path.quadTo(cx, cy + 9.0, cx + span_x, cy - 1.0)
        painter.drawPath(lid_path)

        # Four Minimalist Eyelashes
        lash_pen = QPen(TacticalTheme.EYE_CLOSED_LINE, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(lash_pen)
        lashes = [
            (cx - span_x * 0.72, cy + 2.5, cx - span_x * 0.85, cy + 5.0),
            (cx - span_x * 0.42, cy + 5.0, cx - span_x * 0.52, cy + 8.0),
            (cx + span_x * 0.42, cy + 5.0, cx + span_x * 0.52, cy + 8.0),
            (cx + span_x * 0.72, cy + 2.5, cx + span_x * 0.85, cy + 5.0),
        ]
        for x1, y1, x2, y2 in lashes:
            painter.drawLine(QRectF(x1, y1, x2 - x1, y2 - y1).topLeft(), QRectF(x1, y1, x2 - x1, y2 - y1).bottomRight())

        # Rectangular Top-Right Glint
        glint_w = (w - 12.0) * 0.15
        glint_h = (h - 12.0) * 0.23
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(TacticalTheme.EYE_GLINT))
        painter.drawRect(QRectF(cx + r_shade * 0.32, cy - r_shade * 1.05, glint_w, glint_h))