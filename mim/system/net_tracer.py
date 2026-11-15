import time
import uuid
import socket
import threading
from typing import Optional, Set, Tuple, List, Dict, Any
import psutil
from mim.storage.sqlite_manager import SQLiteManager


class NetworkTracer:
    """
    Monitors active outbound TCP/UDP sockets across the OS and logs new connections
    to the SQLite `network_events` table so timeline scrubbing shows live network activity.
    """

    def __init__(self, sqlite_mgr: SQLiteManager, poll_interval: float = 1.0):
        self.sqlite_mgr = sqlite_mgr
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seen_connections: Set[Tuple[int, str, int, str]] = set()
        self.total_logged = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="MimNetTracer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def poll_once(self) -> int:
        """Polls `psutil.net_connections` once and commits newly established remote endpoints."""
        now_ms = int(time.time() * 1000)
        new_events: List[Dict[str, Any]] = []
        active_set: Set[Tuple[int, str, int, str]] = set()

        try:
            conns = psutil.net_connections(kind="inet")
        except Exception:
            return 0

        for c in conns:
            if not c.raddr or not c.pid:
                continue

            dest_ip = c.raddr.ip
            dest_port = c.raddr.port
            proto = "TCP" if c.type == socket.SOCK_STREAM else "UDP"
            sig = (c.pid, dest_ip, dest_port, proto)
            active_set.add(sig)

            if sig not in self._seen_connections:
                proc_name = "unknown"
                try:
                    proc_name = psutil.Process(c.pid).name()
                except Exception:
                    pass

                new_events.append({
                    "event_id": str(uuid.uuid4()),
                    "timestamp_ms": now_ms,
                    "pid": c.pid,
                    "process_name": proc_name,
                    "protocol": proto,
                    "dest_ip": dest_ip,
                    "dest_port": dest_port,
                    "payload_size": 0,
                })

        self._seen_connections = active_set
        if new_events:
            self.sqlite_mgr.insert_network_events(new_events)
            self.total_logged += len(new_events)

        return len(new_events)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval)