import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Tuple, Set
from mim.config import CONFIG

# Win32 API Constants
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class ProcessGuard:
    """Inspects Win32 foreground windows and blocks ingestion on blacklisted processes or titles."""

    def __init__(
        self,
        blacklisted_processes: Set[str] = CONFIG.blacklisted_processes,
        blacklisted_keywords: Set[str] = CONFIG.blacklisted_title_keywords,
    ):
        self.blacklisted_processes = {p.lower() for p in blacklisted_processes}
        self.blacklisted_keywords = {k.lower() for k in blacklisted_keywords}

        # Bind Win32 DLLs
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

        # Define argtypes/restypes for safety across 64-bit Windows
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD

        self.kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL

    def get_window_metadata(self, hwnd: int) -> Tuple[int, str, str]:
        """Resolves (pid, process_name, window_title) for a given Win32 HWND."""
        if not hwnd:
            return 0, "unknown", ""

        # 1. Resolve Window Title
        length = self.user32.GetWindowTextLengthW(hwnd)
        title = ""
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value

        # 2. Resolve Process ID (PID)
        pid = wintypes.DWORD(0)
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        pid_val = pid.value

        if not pid_val:
            return 0, "unknown", title

        # 3. Resolve Process Executable Name via Kernel32
        process_name = "unknown"
        h_process = self.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid_val)
        if h_process:
            try:
                max_path = 1024
                exe_buf = ctypes.create_unicode_buffer(max_path)
                size = wintypes.DWORD(max_path)
                if self.kernel32.QueryFullProcessImageNameW(h_process, 0, exe_buf, ctypes.byref(size)):
                    full_path = exe_buf.value
                    process_name = Path(full_path).name
            finally:
                self.kernel32.CloseHandle(h_process)

        return pid_val, process_name, title

    def get_active_window_metadata(self) -> Tuple[int, int, str, str]:
        """Returns (hwnd, pid, process_name, window_title) for the current foreground window."""
        hwnd = self.user32.GetForegroundWindow()
        if not hwnd:
            return 0, 0, "unknown", ""
        pid, proc_name, title = self.get_window_metadata(hwnd)
        return int(hwnd), pid, proc_name, title

    def is_context_sensitive(self, process_name: str, window_title: str) -> bool:
        """Evaluates whether a process name or window title triggers the Zero-Trust blocklist."""
        proc_clean = (process_name or "").strip().lower()
        title_clean = (window_title or "").strip().lower()

        if proc_clean in self.blacklisted_processes:
            return True

        for keyword in self.blacklisted_keywords:
            if keyword in title_clean:
                return True

        return False

    def inspect_foreground(self) -> Tuple[bool, int, int, str, str]:
        """
        Inspects the current foreground window.
        Returns: (is_blocked, hwnd, pid, process_name, window_title)
        """
        hwnd, pid, proc_name, title = self.get_active_window_metadata()
        is_blocked = self.is_context_sensitive(proc_name, title)
        return is_blocked, hwnd, pid, proc_name, title