
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
config_path, expected_hash, OUT = (
    Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

assert digest(config_path) == expected_hash
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert config["status"] == "frozen"
assert len(config["scenarios"]) == 22

def check_sources():
    assert digest(config_path) == expected_hash
    for rel, value in config["source_and_input_sha256"].items():
        assert digest(ROOT / rel) == value, rel

check_sources()
E1 = ROOT / config["e1_directory"]
factor_manifest = pd.read_csv(
    ROOT / config["factor_manifest"], dtype={"sessionID": str}
)
methods = ["Uncontrolled", "TOU_Greedy", "RH_LP"]
cohorts = ["main_at_arrival", "supplement_after_submission"]

reference = pd.read_csv(E1 / "comparison.csv")
reference = reference[reference["method"].isin(methods)].copy()
assert len(reference) == 6
reference["scenario_id"] = "E1_reference"
reference["experiment"] = "E4_reference"
reference["amplitude"] = 0.0
reference["seed"] = None
reference["advance_minutes"] = 0
reference["scenario_type"] = "zero_reference_reused"
reference["evidence_directory"] = str(E1.relative_to(ROOT))
frames, outcomes = [reference], []

true_targets = {}
for cohort in cohorts:
    service = pd.read_csv(
        E1 / cohort / "RH_LP/session_service.csv", dtype={"sessionID": str}
    )
    true_targets[cohort] = service.set_index("sessionID")["evaluation_target_kwh"]

for index, scenario in enumerate(config["scenarios"], start=1):
    check_sources()
    folder = OUT / scenario["id"]
    folder.mkdir(exist_ok=False)
    write_json(OUT / "progress.json", {
        "state": "running",
        "current_scenario": scenario["id"],
        "scenario_index": index,
        "scenario_count": 22,
        "completed_count": len(outcomes),
        "outcomes": outcomes,
    })
    command = [
        sys.executable, "-u", str(ROOT / "experiments/run_robustness.py"),
        str(folder), "--split", "test",
        "--alpha", "0.5", "--capacity-kw", "50",
        "--base-hours", "24", "--static-time-limit", "120",
        "--frozen-config-sha256", config["e1_config_sha256"],
        "--protocol-config", str(config_path),
        "--protocol-sha256", expected_hash,
        "--scenario-id", scenario["id"],
    ]
    write_json(folder / "launch.json", {
        "command": command,
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    outcome = {
        "scenario_id": scenario["id"],
        "scenario_type": scenario["type"],
        "status": "failed",
    }
    try:
        with (folder / "stdout_stderr.txt").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        outcome["returncode"] = process.returncode
        write_json(folder / "process_exit.json", {
            "returncode": process.returncode
        })
        assert process.returncode == 0, "Scenario process failed"
        record = json.loads(
            (folder / "verification.json").read_text(encoding="utf-8")
        )
        assert record["status"] == "passed"
        for rel, value in record["source_and_input_sha256"].items():
            assert digest(ROOT / rel) == value, rel
        for rel, value in record["output_artifact_sha256"].items():
            assert digest(folder / rel) == value, rel

        frame = pd.read_csv(folder / "comparison.csv")
        assert len(frame) == 6
        assert set(zip(frame["cohort"], frame["method"])) == {
            (c, m) for c in cohorts for m in methods
        }
        for cohort in cohorts:
            expected = factor_manifest[
                (factor_manifest["scenario_id"] == scenario["id"]) &
                (factor_manifest["cohort"] == cohort)
            ].set_index("sessionID")["factor"].sort_index()
            actual = pd.read_csv(
                folder / cohort / "request_factors.csv",
                dtype={"sessionID": str}
            ).set_index("sessionID")["factor"].sort_index()
            assert actual.index.equals(expected.index)
            assert np.max(np.abs(
                actual.to_numpy() - expected.to_numpy()
            )) <= 1e-12

            targets = true_targets[cohort].sort_index()
            for method in methods:
                service = pd.read_csv(
                    folder / cohort / method / "session_service.csv",
                    dtype={"sessionID": str}
                )
                current = service.set_index(
                    "sessionID"
                )["evaluation_target_kwh"].sort_index()
                assert current.index.equals(targets.index)
                assert np.max(np.abs(
                    current.to_numpy() - targets.to_numpy()
                )) <= 1e-8

        check_sources()
        frame["scenario_type"] = scenario["type"]
        frame["evidence_directory"] = str(folder.relative_to(ROOT))
        frames.append(frame)
        outcome.update(
            status=("requires_review" if record["requires_review"] else "passed"),
            check_count=record["check_count"],
            wall_seconds=record["wall_seconds"],
        )
    except Exception:
        outcome["error"] = traceback.format_exc()

    outcomes.append(outcome)
    write_json(folder / "scenario_outcome.json", outcome)
    write_json(OUT / "outcomes.json", outcomes)
    pd.concat(frames, ignore_index=True).to_csv(
        OUT / "robustness_results.csv", index=False
    )
    print(json.dumps(outcome, ensure_ascii=False), flush=True)

check_sources()
passed = all(item["status"] == "passed" for item in outcomes)
result = {
    "status": "passed" if passed else "requires_review",
    "scope": "E4 prescribed 22 scenarios; E1 zero reference reused",
    "config_sha256": expected_hash,
    "scenarios": outcomes,
    "reference_directory": str(E1.relative_to(ROOT)),
    "reference_policy": (
        "E1 online runs reused; zero-transform equivalence was checked "
        "on a small sample, not by a full zero-error rerun."
    ),
    "results_sha256": digest(OUT / "robustness_results.csv"),
}
write_json(OUT / "batch_verification.json", result)
write_json(OUT / "progress.json", {
    "state": "finished", "completed_count": len(outcomes),
    "outcomes": outcomes,
})
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if passed else 1)
