import numpy as np

from ueba_prototype.collector import TelemetrySample
from ueba_prototype.reporter import build_report


def test_build_report_contains_security_explanation() -> None:
    sample = TelemetrySample(
        timestamp=1.0,
        hostname="host",
        os_name="Windows",
        interval_seconds=5.0,
        cpu_percent=10.0,
        memory_percent=50.0,
        swap_percent=0.0,
        process_count=100,
        user_process_count=80,
        new_process_count=1,
        thread_count=900,
        open_file_count=50,
        connection_count=200,
        established_count=150,
        listen_count=10,
        remote_ip_count=60,
        remote_port_count=30,
        uncommon_remote_port_count=20,
        bytes_sent_per_sec=100000.0,
        bytes_recv_per_sec=1000.0,
        packets_sent_per_sec=300.0,
        packets_recv_per_sec=20.0,
    )
    columns = ["connection_count", "bytes_sent_per_sec", "new_process_count"]
    normalized = np.array([[10.0, 0.0, 0.0]])
    reconstructed = np.array([[0.0, 0.0, 0.0]])

    report = build_report(sample, columns, normalized, reconstructed, anomaly_score=100.0, threshold=10.0)

    assert report.top_features[0]["feature"] == "connection_count"
    assert any("network connections" in explanation for explanation in report.explanations)
