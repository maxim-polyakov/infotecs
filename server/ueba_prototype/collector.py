from __future__ import annotations

import csv
import platform
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import psutil


COMMON_REMOTE_PORTS = {20, 21, 22, 25, 53, 80, 110, 123, 143, 389, 443, 445, 465, 587, 993, 995, 3389}


@dataclass
class TelemetrySample:
    timestamp: float
    hostname: str
    os_name: str
    interval_seconds: float
    cpu_percent: float
    memory_percent: float
    swap_percent: float
    process_count: int
    user_process_count: int
    new_process_count: int
    thread_count: int
    open_file_count: int
    connection_count: int
    established_count: int
    listen_count: int
    remote_ip_count: int
    remote_port_count: int
    uncommon_remote_port_count: int
    bytes_sent_per_sec: float
    bytes_recv_per_sec: float
    packets_sent_per_sec: float
    packets_recv_per_sec: float


class TelemetryCollector:
    """Collects local host behavior using APIs available on Windows and Linux."""

    def __init__(self) -> None:
        self._previous_pids: set[int] = set(psutil.pids())
        self._previous_net = psutil.net_io_counters()
        self._previous_timestamp = time.time()
        psutil.cpu_percent(interval=None)

    def sample(self, interval_seconds: float | None = None) -> TelemetrySample:
        now = time.time()
        elapsed = interval_seconds or max(now - self._previous_timestamp, 1.0)

        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        process_count = 0
        user_process_count = 0
        thread_count = 0
        open_file_count = 0
        current_pids: set[int] = set()

        for proc in psutil.process_iter(attrs=["pid", "username", "num_threads"]):
            process_count += 1
            info = proc.info
            pid = int(info.get("pid") or 0)
            current_pids.add(pid)
            if info.get("username"):
                user_process_count += 1
            thread_count += int(info.get("num_threads") or 0)
            try:
                open_file_count += len(proc.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue

        new_process_count = len(current_pids - self._previous_pids)
        self._previous_pids = current_pids

        connection_stats = self._connection_stats()
        net = psutil.net_io_counters()
        sent_delta = max(net.bytes_sent - self._previous_net.bytes_sent, 0)
        recv_delta = max(net.bytes_recv - self._previous_net.bytes_recv, 0)
        packets_sent_delta = max(net.packets_sent - self._previous_net.packets_sent, 0)
        packets_recv_delta = max(net.packets_recv - self._previous_net.packets_recv, 0)
        self._previous_net = net
        self._previous_timestamp = now

        return TelemetrySample(
            timestamp=now,
            hostname=socket.gethostname(),
            os_name=platform.system(),
            interval_seconds=elapsed,
            cpu_percent=float(cpu_percent),
            memory_percent=float(memory.percent),
            swap_percent=float(swap.percent),
            process_count=process_count,
            user_process_count=user_process_count,
            new_process_count=new_process_count,
            thread_count=thread_count,
            open_file_count=open_file_count,
            bytes_sent_per_sec=sent_delta / elapsed,
            bytes_recv_per_sec=recv_delta / elapsed,
            packets_sent_per_sec=packets_sent_delta / elapsed,
            packets_recv_per_sec=packets_recv_delta / elapsed,
            **connection_stats,
        )

    def _connection_stats(self) -> dict[str, int]:
        remote_ips: set[str] = set()
        remote_ports: set[int] = set()
        connection_count = 0
        established_count = 0
        listen_count = 0

        try:
            connections = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, OSError):
            connections = []

        for conn in connections:
            connection_count += 1
            if conn.status == psutil.CONN_ESTABLISHED:
                established_count += 1
            elif conn.status == psutil.CONN_LISTEN:
                listen_count += 1

            if conn.raddr:
                remote_ips.add(str(conn.raddr.ip))
                remote_ports.add(int(conn.raddr.port))

        uncommon_ports = {port for port in remote_ports if port not in COMMON_REMOTE_PORTS}
        return {
            "connection_count": connection_count,
            "established_count": established_count,
            "listen_count": listen_count,
            "remote_ip_count": len(remote_ips),
            "remote_port_count": len(remote_ports),
            "uncommon_remote_port_count": len(uncommon_ports),
        }


def append_samples(path: Path, samples: Iterable[TelemetrySample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(sample) for sample in samples]
    if not rows:
        return 0

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def collect_to_csv(path: Path, duration_seconds: float, interval_seconds: float) -> int:
    collector = TelemetryCollector()
    deadline = time.time() + duration_seconds
    written = 0

    while time.time() < deadline:
        time.sleep(interval_seconds)
        sample = collector.sample(interval_seconds=interval_seconds)
        written += append_samples(path, [sample])

    return written
