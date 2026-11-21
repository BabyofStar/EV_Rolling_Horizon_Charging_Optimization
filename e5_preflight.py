"""Collect the source evidence needed to implement E5 without running a replay."""

import argparse
import ast
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REQUIRED = (
    "src/rolling_horizon.py",
    "src/simulator.py",
    "src/acnsim_adapter.py",
    "src/optimization_model.py",
    "src/tariff.py",
    "src/metrics.py",
    "experiments/run_rolling.py",
    "experiments/run_evaluation.py",
)

OPTIONAL = (
    "src/baselines.py",
    "experiments/run_all.py",
    "experiments/run_frozen_batch.py",
    "tests/test_rolling_horizon.py",
    "tests/test_static_lp.py",
    "tests/test_metrics.py",
    "tests/test_trajectory_timestamps.py",
)

PATTERNS = (
    ("grid", r"interval_minutes|\b(?:step|slot|grid)_minutes\b|\bdt_hours\b|"
     r"15\s*min|minutes\s*=\s*15\b|(?<![\w.])0\.25\b|\.ceil\(|\.floor\("),
    ("horizon", r"base_hours|base_horizon|effective_horizon|deadline|overdue"),
    ("timing", r"perf_counter|solver_seconds|solve_seconds|total_seconds|time_limit|"
     r"wall_seconds|fallback|num_variables|num_constraints"),
    ("pricing", r"price_at|attach_tariff|tariff_lookup"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    project = args.project.resolve()

    missing = [name for name in REQUIRED if not (project / name).is_file()]
    if missing:
        parser.error(f"Missing expected project files under {project}: {', '.join(missing)}")

    report = [f"Python {sys.version.split()[0]}", "E5 source preflight", ""]
    source_files = {}
    for name in REQUIRED + OPTIONAL:
        path = project / name
        if not path.is_file():
            report.append(f"Optional file absent: {name}")
            continue

        data = path.read_bytes()
        source = data.decode("utf-8-sig")
        ast.parse(source, filename=name)
        source_files[name] = data
        report.extend(("", f"{name}  sha256={hashlib.sha256(data).hexdigest()}"))
        for number, line in enumerate(source.splitlines(), 1):
            categories = [label for label, pattern in PATTERNS if re.search(pattern, line, re.I)]
            if categories:
                report.append(f"  {number:>4} [{', '.join(categories)}] {line.strip()}")

    # Report filenames only. Configuration files and experiment data stay on the server.
    config_names = sorted(
        path.name for path in (project / "configs").glob("*.yaml") if path.is_file()
    )
    report.extend(("", "YAML configuration filenames: " + ", ".join(config_names), ""))
    report_text = "\n".join(report)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = project / "results" / "logs" / f"e5_preflight_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "e5_preflight.txt"
    report_path.write_text(report_text, encoding="utf-8")
    archive_path = output_dir / "e5_preflight_sources.zip"
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in source_files.items():
            archive.writestr(name, data)
        archive.writestr(report_path.name, report_text)

    print(f"Preflight collected {len(source_files)} Python files.")
    print(f"Report: {report_path.relative_to(project)}")
    print(f"Source archive: {archive_path.relative_to(project)}")


if __name__ == "__main__":
    main()
