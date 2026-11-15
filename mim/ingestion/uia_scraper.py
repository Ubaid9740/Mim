import uuid
from typing import List, Dict, Any, Optional
import uiautomation as auto
from mim.config import CONFIG
from mim.security.sanitizer import ZeroTrustSanitizer


class UIAScraper:
    """
    Walks the Windows UI Automation (UIA) accessibility tree for the active HWND,
    extracting bounding boxes, control types, and sanitized text strings.
    """

    def __init__(
        self,
        sanitizer: ZeroTrustSanitizer,
        max_depth: int = CONFIG.uia_max_depth,
        max_nodes_per_frame: int = 250,
    ):
        self.sanitizer = sanitizer
        self.max_depth = max_depth
        self.max_nodes_per_frame = max_nodes_per_frame

    def _extract_control_text(self, control: auto.Control, process_name: str) -> str:
        """Extracts Name, ValuePattern, or LegacyIAccessible text and sanitizes it."""
        pieces = []

        try:
            name = (control.Name or "").strip()
            if name:
                pieces.append(name)
        except Exception:
            pass

        # Attempt ValuePattern extraction (e.g., URL bars, input fields, code editors)
        try:
            val_pattern = control.GetValuePattern()
            if val_pattern:
                val = (val_pattern.Value or "").strip()
                if val and val not in pieces:
                    pieces.append(val)
        except Exception:
            pass

        if not pieces:
            return ""

        combined = " | ".join(pieces)
        # Enforce a reasonable per-node character ceiling before AST/Regex sanitization
        if len(combined) > 4000:
            combined = combined[:4000]

        return self.sanitizer.sanitize(combined, process_name=process_name)

    def scrape_window(self, hwnd: int, process_name: str = "") -> List[Dict[str, Any]]:
        """
        Walks the UIA tree starting from `hwnd` up to `self.max_depth`.
        Must be invoked inside a `with auto.UIAutomationInitializerInThread():` block
        when running on a background worker thread.
        """
        if not hwnd:
            return []

        nodes: List[Dict[str, Any]] = []

        try:
            root_control = auto.ControlFromHandle(hwnd)
            if not root_control:
                return []
        except Exception:
            return []

        def _walk(
            ctrl: auto.Control,
            parent_id: Optional[str],
            depth: int
        ) -> None:
            if depth > self.max_depth or len(nodes) >= self.max_nodes_per_frame:
                return

            try:
                if ctrl.IsOffscreen:
                    return
            except Exception:
                return

            # Extract BoundingRectangle
            bbox = [0, 0, 0, 0]
            try:
                rect = ctrl.BoundingRectangle
                if rect:
                    bbox = [int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)]
            except Exception:
                pass

            # Ignore degenerate 0-area invisible nodes unless it is the root window
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if depth > 0 and (width <= 1 or height <= 1):
                return

            try:
                ctrl_type = int(ctrl.ControlTypeId)
            except Exception:
                ctrl_type = 0

            try:
                is_enabled = bool(ctrl.IsEnabled)
            except Exception:
                is_enabled = True

            sanitized_text = self._extract_control_text(ctrl, process_name=process_name)
            node_id = str(uuid.uuid4())

            # Keep nodes that either carry text or represent meaningful interactive controls
            if sanitized_text or depth <= 2:
                nodes.append({
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "control_type": ctrl_type,
                    "bounding_box": bbox,
                    "is_interactable": is_enabled,
                    "content_text": sanitized_text,
                    "source_type": "UIA",
                })
                current_parent_id = node_id
            else:
                current_parent_id = parent_id

            if depth == self.max_depth or len(nodes) >= self.max_nodes_per_frame:
                return

            # Traverse child controls safely
            try:
                children = ctrl.GetChildren()
            except Exception:
                children = []

            for child in children:
                if len(nodes) >= self.max_nodes_per_frame:
                    break
                _walk(child, current_parent_id, depth + 1)

        _walk(root_control, None, 0)
        return nodes