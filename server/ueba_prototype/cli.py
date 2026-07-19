from __future__ import annotations

import argparse
from pathlib import Path

from .collector import collect_to_csv
from .demo import run_demo
from .detector import run_detection_loop
from .features import FEATURE_COLUMNS, load_csv, rows_to_matrix
from .model import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UEBA prototype for local PC behavior anomaly detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect training telemetry")
    collect.add_argument("--duration-hours", type=float, default=24.0, help="collection duration; use 24+ for final training")
    collect.add_argument("--interval", type=float, default=5.0, help="sampling interval in seconds")
    collect.add_argument("--out", type=Path, default=Path("data/raw.csv"), help="CSV output path")

    train = subparsers.add_parser("train", help="train the autoencoder model")
    train.add_argument("--data", type=Path, required=True, help="CSV collected by the collect command")
    train.add_argument("--model-dir", type=Path, default=Path("models/default"), help="directory for saved model files")
    train.add_argument("--epochs", type=int, default=250, help="training epochs")
    train.add_argument("--learning-rate", type=float, default=0.01, help="gradient descent learning rate")
    train.add_argument("--threshold-quantile", type=float, default=0.995, help="training error quantile used as anomaly threshold")

    detect = subparsers.add_parser("detect", help="run continuous anomaly detection")
    detect.add_argument("--model-dir", type=Path, default=Path("models/default"), help="directory with saved model files")
    detect.add_argument("--interval", type=float, default=5.0, help="sampling interval in seconds")
    detect.add_argument("--reports", type=Path, default=Path("reports/anomalies.jsonl"), help="JSONL anomaly report path")
    detect.add_argument("--max-samples", type=int, default=None, help="optional limit for test runs")

    demo = subparsers.add_parser("demo", help="train and detect anomalies on synthetic data")
    demo.add_argument("--output-dir", type=Path, default=Path("reports/demo"), help="directory for demo artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "collect":
        duration_seconds = args.duration_hours * 60 * 60
        written = collect_to_csv(args.out, duration_seconds=duration_seconds, interval_seconds=args.interval)
        print(f"Collected {written} telemetry samples into {args.out}")
        return 0

    if args.command == "train":
        rows = load_csv(args.data)
        matrix = rows_to_matrix(rows, FEATURE_COLUMNS)
        bundle = train_model(
            matrix,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold_quantile=args.threshold_quantile,
        )
        bundle.save(args.model_dir)
        print(f"Saved model to {args.model_dir}")
        print(f"Anomaly threshold: {bundle.threshold:.6f}")
        return 0

    if args.command == "detect":
        def _print_report(report: object) -> None:
            score = getattr(report, "anomaly_score")
            threshold = getattr(report, "threshold")
            explanations = "; ".join(getattr(report, "explanations"))
            print(f"Anomaly detected: score={score:.6f}, threshold={threshold:.6f}, {explanations}")

        analyzed = run_detection_loop(
            model_dir=args.model_dir,
            report_path=args.reports,
            interval_seconds=args.interval,
            max_samples=args.max_samples,
            on_report=_print_report,
        )
        print(f"Analyzed {analyzed} samples; anomaly reports are stored in {args.reports}")
        return 0

    if args.command == "demo":
        reports = run_demo(args.output_dir)
        print(f"Demo completed with {len(reports)} detected anomalies")
        print(f"Artifacts: {args.output_dir}")
        for report in reports:
            top = ", ".join(str(item["feature"]) for item in report.top_features[:3])
            print(f"- score={report.anomaly_score:.6f}, top_features={top}")
        return 0

    return 1
