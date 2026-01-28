from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RunSummary:
    run_dir: Path
    condition: str
    window: int
    model: str
    fusion: str
    modality: str
    metric_key: str
    mean: float
    std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation CSV results.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory containing run subfolders.",
    )
    parser.add_argument(
        "--metric",
        default="r2",
        help="Metric prefix to summarize (e.g., r2, rmse_raw, mae_raw).",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"],
        help="Targets to include in summary.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def iter_metric_files(runs_dir: Path) -> Iterable[Path]:
    for path in runs_dir.rglob("mc_metrics_*.csv"):
        yield path


def parse_run_dir(run_dir: Path) -> tuple[str, str, str, int, str]:
    name = run_dir.name
    parts = name.split("_")
    model = "unknown"
    fusion = "unknown"
    modality = "unknown"
    condition = "unknown"
    window = -1
    if len(parts) >= 6:
        model = parts[2]
        fusion = parts[3]
        modality = parts[4]
        for part in parts[5:]:
            if part.lower() in {"light", "dark"}:
                condition = part
            if part.startswith("w"):
                try:
                    window = int(part[1:])
                except ValueError:
                    pass
    return model, fusion, modality, window, condition


def load_metric_values(path: Path, key: str) -> list[float]:
    values = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if key in row and row[key] != "":
                values.append(float(row[key]))
    return values


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def collect_summaries(runs_dir: Path, metric: str, targets: list[str]) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for csv_path in iter_metric_files(runs_dir):
        run_dir = csv_path.parent
        model, fusion, modality, window, condition = parse_run_dir(run_dir)
        for target in targets:
            metric_key = f"{metric}_{target}"
            values = load_metric_values(csv_path, metric_key)
            mean, std = summarize(values)
            summaries.append(
                RunSummary(
                    run_dir=run_dir,
                    condition=condition,
                    window=window,
                    model=model,
                    fusion=fusion,
                    modality=modality,
                    metric_key=metric_key,
                    mean=mean,
                    std=std,
                )
            )
    return summaries


def render_markdown(rows: list[RunSummary]) -> str:
    lines = []
    lines.append("| Condition | Window | Model | Fusion | Modality | Metric | Mean | Std | Run Dir |")
    lines.append("| :--- | :---: | :--- | :--- | :--- | :--- | ---: | ---: | :--- |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.condition,
                    str(row.window),
                    row.model,
                    row.fusion,
                    row.modality,
                    row.metric_key,
                    f"{row.mean:.4f}",
                    f"{row.std:.4f}",
                    row.run_dir.name,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_csv(rows: list[RunSummary]) -> str:
    header = ["condition", "window", "model", "fusion", "modality", "metric", "mean", "std", "run_dir"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row.condition,
                    str(row.window),
                    row.model,
                    row.fusion,
                    row.modality,
                    row.metric_key,
                    f"{row.mean:.6f}",
                    f"{row.std:.6f}",
                    row.run_dir.name,
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = collect_summaries(runs_dir, args.metric, args.targets)
    rows = sorted(rows, key=lambda r: (r.condition, r.window, r.model, r.metric_key))
    if args.format == "markdown":
        output = render_markdown(rows)
    else:
        output = render_csv(rows)
    print(output)


if __name__ == "__main__":
    main()
