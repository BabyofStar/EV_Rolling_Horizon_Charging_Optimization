
"""Independent clean-environment smoke and frozen-summary reproduction."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import venv

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def clean_env():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE",
                "PIP_USER", "PIP_TARGET", "PIP_PREFIX"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env

def run(command, cwd, log, timeout):
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(v) for v in command], cwd=cwd, env=clean_env(),
            stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"Child exit {result.returncode}; inspect {log}")

def worker(out):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pandas.testing import assert_frame_equal

    out.mkdir(parents=True, exist_ok=False)
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Worker is not running in a virtual environment")
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if "include-system-site-packages = false" not in cfg.read_text().lower():
        raise RuntimeError("Environment permits inherited system packages")

    run([sys.executable, ROOT / "tests/test_static_lp.py",
         "--output", out / "static_cases"],
        ROOT, out / "static.stdout.txt", 600)
    static = json.loads(
        (out / "static_cases/validation_summary.json").read_text())
    if static["status"] != "passed" or (
        static["passed"], static["total"]) != (16, 16):
        raise RuntimeError("Static synthetic validation did not pass 16/16")

    run([sys.executable, ROOT / "tests/test_rolling_horizon.py",
         out / "rolling.json"],
        ROOT, out / "rolling.stdout.txt", 600)
    rolling = json.loads((out / "rolling.json").read_text())
    if rolling["status"] != "passed" or rolling["check_count"] != 17:
        raise RuntimeError("RH synthetic validation did not pass 17/17")

    source = ROOT / "results/stage4_all_comparisons.csv"
    wanted = "1096d28710fe444a0e1f7003bcc5cb6c67878b6983dbab2d1f8fbdecef405e8f"
    if digest(source) != wanted:
        raise RuntimeError("Frozen results fingerprint mismatch")
    df = pd.read_csv(source)
    if len(df) != 172:
        raise RuntimeError("Unexpected result count")
    e1 = df.loc[df.experiment.eq("E1")].copy()
    reference = e1.loc[e1.method.ne("Static_LP"), [
        "cohort", "method", "cost_USD", "unmet_kwh", "energy_satisfaction"
    ]].rename(columns={
        "cost_USD": "reference_cost_USD",
        "unmet_kwh": "reference_unmet_kwh",
        "energy_satisfaction": "reference_satisfaction",
    })
    keys = ["cohort", "method"]
    group = keys + ["amplitude"]
    noise = df.loc[
        df.experiment.eq("E4") &
        pd.to_numeric(df.amplitude, errors="coerce").gt(0)
    ].copy()
    if len(noise) != 120 or noise.duplicated(group + ["seed"]).any():
        raise RuntimeError("Invalid seed grid")
    stats = noise.groupby(group, as_index=False).agg(
        seed_count=("seed", "nunique"),
        cost_mean_USD=("cost_USD", "mean"),
        unmet_mean_kwh=("unmet_kwh", "mean"),
        excess_mean_kwh=("excess_kwh", "mean"),
        satisfaction_mean=("energy_satisfaction", "mean"),
        satisfaction_SD=("energy_satisfaction", "std"),
    )
    if len(stats) != 12 or not stats.seed_count.eq(10).all():
        raise RuntimeError("Expected twelve groups of ten seeds")
    stats = stats.merge(reference, on=keys, validate="many_to_one")
    stats["satisfaction_mean_pct"] = stats.satisfaction_mean * 100
    stats["satisfaction_SD_pp"] = stats.satisfaction_SD * 100
    stats["delta_satisfaction_pp"] = (
        stats.satisfaction_mean - stats.reference_satisfaction) * 100
    stats["delta_unmet_kwh"] = (
        stats.unmet_mean_kwh - stats.reference_unmet_kwh)

    early = df.loc[
        df.experiment.eq("E4") &
        pd.to_numeric(df.advance_minutes, errors="coerce").gt(0)
    ].copy().merge(reference, on=keys, validate="many_to_one")
    early["delta_satisfaction_pp"] = (
        early.energy_satisfaction - early.reference_satisfaction) * 100
    early["delta_unmet_kwh"] = early.unmet_kwh - early.reference_unmet_kwh

    tables = {
        "sensitivity_all_methods.csv":
            df.loc[df.experiment.isin(["E1", "E2", "E3"])].copy(),
        "demand_error_seed_statistics.csv": stats,
        "early_departure_comparisons.csv": early,
    }
    table_checks = {}
    for name, frame in tables.items():
        target = out / name
        frame.to_csv(target, index=False)
        # 比较重新读入的数值，避免CSV末位格式差异造成误报
        actual = pd.read_csv(target)
        expected = pd.read_csv(ROOT / "report/tables" / name)
        assert_frame_equal(
            actual, expected, check_dtype=False, check_exact=False,
            rtol=1e-10, atol=1e-8)
        table_checks[name] = {
            "status": "passed", "rows": len(frame), "sha256": digest(target)}

    # 独立汇总图：验证干净环境能从冻结结果生成图表
    methods = ["Uncontrolled", "TOU_Greedy", "Static_LP", "RH_LP"]
    cohorts = ["main_at_arrival", "supplement_after_submission"]
    colors = ["#64748b", "#f59e0b", "#059669", "#2563eb"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, cohort in enumerate(cohorts):
        part = e1.loc[e1.cohort.eq(cohort)].set_index("method").loc[methods]
        for j, (metric, label, scale) in enumerate([
            ("cost_USD", "Cost (USD)", 1),
            ("energy_satisfaction", "Energy satisfaction (%)", 100),
            ("max_peak_kW", "Maximum EV power (kW)", 1),
        ]):
            ax = axes[i, j]
            ax.bar(["UC", "TOU", "Static", "RH"], part[metric] * scale,
                   color=colors)
            ax.set_title(cohort.replace("_", " "))
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen E1 summary; Static is an offline reference")
    fig.savefig(out / "E1_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, cohort in zip(axes, cohorts):
        for method, color in zip(
            ["Uncontrolled", "TOU_Greedy", "RH_LP"],
            [colors[0], colors[1], colors[3]]
        ):
            part = stats.loc[
                stats.cohort.eq(cohort) & stats.method.eq(method)
            ].sort_values("amplitude")
            base = reference.loc[
                reference.cohort.eq(cohort) & reference.method.eq(method)
            ].iloc[0]
            x = [0] + (part.amplitude * 100).tolist()
            y = [base.reference_satisfaction * 100] + (
                part.satisfaction_mean_pct.tolist())
            error = [0] + part.satisfaction_SD_pp.tolist()
            ax.errorbar(x, y, yerr=error, marker="o",
                        capsize=3, label=method, color=color)
        ax.set_title(cohort.replace("_", " "))
        ax.set_xlabel("Controlled error amplitude (%)")
        ax.set_ylabel("Energy satisfaction (%)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Mean +/- sample SD over 10 seeds; zero reuses E1")
    fig.savefig(out / "demand_error_summary.png", dpi=180)
    plt.close(fig)

    result = {
        "status": "passed",
        "scope": "clean synthetic static/RH/ACN-Sim and frozen-result summary",
        "static_checks": static["passed"],
        "rolling_checks": rolling["check_count"],
        "table_checks": table_checks,
        "generated_figures": ["E1_summary.png", "demand_error_summary.png"],
        "python": sys.version,
        "system_site_packages": False,
        "full_formal_experiments_rerun": False,
        "raw_data_preprocessing_rerun": False,
    }
    (out / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

def supervisor(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise FileExistsError("Use a new output directory")
    started = time.perf_counter()

    def state(status, stage, **extra):
        value = {
            "status": status, "stage": stage,
            "elapsed_seconds": time.perf_counter() - started, **extra}
        temporary = out / "status.tmp"
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(out / "status.json")

    try:
        state("running", "copying_workspace")
        snapshot = out / "workspace"
        snapshot.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for folder in ("src", "configs", "tests"):
            shutil.copytree(ROOT / folder, snapshot / folder, ignore=ignore)
        files = [
            "experiments/reproduce_stage5.py", "requirements.reproduce.txt",
            "results/stage4_all_comparisons.csv",
            "report/tables/sensitivity_all_methods.csv",
            "report/tables/demand_error_seed_statistics.csv",
            "report/tables/early_departure_comparisons.csv",
        ]
        snapshots = {}
        for rel in files:
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
            snapshots[rel] = digest(target)
        (out / "snapshot_files.json").write_text(
            json.dumps(snapshots, indent=2), encoding="utf-8")

        envdir = ROOT / ".envs" / ("reproduce-" + out.name)
        if envdir.exists():
            raise FileExistsError(envdir)
        state("running", "creating_virtual_environment")
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(envdir)
        python = envdir / "bin/python"

        state("running", "installing_dependencies")
        run([python, "-m", "pip", "--disable-pip-version-check",
             "install", "-r", snapshot / "requirements.reproduce.txt"],
            snapshot, out / "install.log", 1800)
        run([python, "-m", "pip", "check"],
            snapshot, out / "pip_check.log", 120)

        state("running", "running_synthetic_checks_and_summary")
        evidence = snapshot / "results/reproduction"
        run([python, snapshot / "experiments/reproduce_stage5.py",
             "--worker", "--output", evidence],
            snapshot, out / "worker.log", 1800)
        result = json.loads((evidence / "verification.json").read_text())
        if result.get("status") != "passed":
            raise RuntimeError("Worker did not pass")
        state("passed", "complete", evidence=str(evidence),
              environment=str(envdir), result=result)
    except Exception as exc:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state("failed", "stopped", error=repr(exc))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.worker:
        worker(output)
    else:
        supervisor(output)
