import ctypes
from ctypes import wintypes
from PyQt6.QtGui import QColor

# Win32 Display Affinity Constant: Excludes window from DXGI/MSS screen capture
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class TacticalTheme:
    """
    Tactical Observability (Flight Recorder) Color System & QSS Tokens:
    - Base: Pitch Black (#000000) at 70% opacity -> rgba(0, 0, 0, 178)
    - Active/Recording State: Terminal Amber (#FFB000)
    - Safe/Paused State: Ghost White (#F3F3F3)
    - UIA Bounding Boxes: Matrix Cyan (#00E5FF)
    """

    # Hex Strings
    HEX_PITCH_BLACK = "#000000"
    HEX_TERMINAL_AMBER = "#FFB000"
    HEX_GHOST_WHITE = "#F3F3F3"
    HEX_MATRIX_CYAN = "#00E5FF"
    HEX_ALERT_RED = "#FF3B30"
    HEX_PANEL_BORDER = "#2A2E35"

    # QColor Instances
    PITCH_BLACK_70 = QColor(0, 0, 0, 178)
    PITCH_BLACK_88 = QColor(8, 10, 14, 225)
    TERMINAL_AMBER = QColor(255, 176, 0, 255)
    TERMINAL_AMBER_GLOW = QColor(255, 176, 0, 75)
    GHOST_WHITE = QColor(243, 243, 243, 255)
    GHOST_WHITE_DIM = QColor(243, 243, 243, 140)
    MATRIX_CYAN = QColor(0, 229, 255, 255)
    MATRIX_CYAN_FILL = QColor(0, 229, 255, 42)
    MATRIX_CYAN_SUBTLE = QColor(0, 229, 255, 95)

    # Eye Sentinel Artwork Palette (Matches user-supplied Open & Closed Eye Reference)
    EYE_SCLERA_OUTER = QColor(225, 228, 232, 255)
    EYE_SCLERA_INNER = QColor(248, 249, 250, 255)
    EYE_IRIS_OUTER = QColor(74, 114, 184, 255)
    EYE_IRIS_INNER = QColor(56, 91, 156, 255)
    EYE_PUPIL = QColor(21, 31, 48, 255)
    EYE_GLINT = QColor(255, 255, 255, 215)
    EYE_CLOSED_LINE = QColor(24, 30, 44, 255)

    # PyQt6 Global Stylesheet for Scrubber Bar & Telemetry Panels
    OVERLAY_STYLESHEET = """
        QWidget#ScrubberBar, QFrame#TelemetryPanel {
            background-color: rgba(0, 0, 0, 195);
            border: 1px solid rgba(0, 229, 255, 90);
            border-radius: 10px;
            color: #F3F3F3;
            font-family: 'Cascadia Code', 'Consolas', 'Segoe UI', monospace;
        }
        QLabel#StateBadgeActive {
            background-color: rgba(255, 176, 0, 35);
            color: #FFB000;
            border: 1px solid #FFB000;
            border-radius: 5px;
            padding: 4px 8px;
            font-weight: bold;
            font-size: 11px;
        }
        QLabel#StateBadgePaused {
            background-color: rgba(243, 243, 243, 25);
            color: #F3F3F3;
            border: 1px solid #F3F3F3;
            border-radius: 5px;
            padding: 4px 8px;
            font-weight: bold;
            font-size: 11px;
        }
        QLabel#ProcessBadge {
            color: #00E5FF;
            font-weight: bold;
            font-size: 12px;
        }
        QLabel#MetaText {
            color: #F3F3F3;
            font-size: 11px;
        }
        QLineEdit#SearchInput {
            background-color: rgba(18, 22, 28, 220);
            color: #F3F3F3;
            border: 1px solid rgba(243, 243, 243, 80);
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 12px;
        }
        QLineEdit#SearchInput:focus {
            border: 1px solid #00E5FF;
        }
        QPushButton#ActionBtn {
            background-color: rgba(0, 229, 255, 25);
            color: #00E5FF;
            border: 1px solid #00E5FF;
            border-radius: 6px;
            padding: 5px 12px;
            font-weight: bold;
            font-size: 11px;
        }
        QPushButton#ActionBtn:hover {
            background-color: rgba(0, 229, 255, 65);
        }
        QPushButton#RollbackBtn {
            background-color: rgba(255, 176, 0, 30);
            color: #FFB000;
            border: 1px solid #FFB000;
            border-radius: 6px;
            padding: 5px 12px;
            font-weight: bold;
            font-size: 11px;
        }
        QPushButton#RollbackBtn:hover {
            background-color: rgba(255, 176, 0, 75);
        }
        QSlider::groove:horizontal {
            border: 1px solid rgba(243, 243, 243, 60);
            height: 6px;
            background: rgba(20, 24, 30, 200);
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: rgba(0, 229, 255, 150);
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #FFB000;
            border: 2px solid #000000;
            width: 16px;
            margin-top: -6px;
            margin-bottom: -6px;
            border-radius: 8px;
        }
        QListWidget#TelemetryList {
            background-color: rgba(10, 12, 16, 180);
            color: #F3F3F3;
            border: 1px solid rgba(243, 243, 243, 40);
            border-radius: 6px;
            font-size: 11px;
        }
    """


def apply_window_capture_cloaking(hwnd: int) -> bool:
    """
    Applies Win32 `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`
    so Mim's DXGI/MSS screen grabber never records its own UI widgets.
    """
    if not hwnd:
        return False
    try:
        user32 = ctypes.windll.user32
        user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
        res = user32.SetWindowDisplayAffinity(wintypes.HWND(int(hwnd)), WDA_EXCLUDEFROMCAPTURE)
        return bool(res)
    except Exception:
        return False