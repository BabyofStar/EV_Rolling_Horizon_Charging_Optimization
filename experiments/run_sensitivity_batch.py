
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import traceback
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
design_path, expected_hash, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

assert digest(design_path) == expected_hash
design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
assert design["status"] == "frozen"
for rel, value in design["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == value, rel

base = ROOT / design["e1_directory"]
base_v = json.loads((base / "verification.json").read_text(encoding="utf-8"))
for rel, value in base_v["output_artifact_sha256"].items():
    assert digest(base / rel) == value, rel

reference = pd.read_csv(base / "comparison.csv")
reference["scenario_id"] = "E1_reference"
reference["experiment"] = "shared_E2_E3_reference"
reference["evidence_directory"] = str(base.relative_to(ROOT))
frames, outcomes = [reference], []

for scenario in design["scenarios"]:
    folder = output / scenario["id"]
    folder.mkdir(exist_ok=False)
    write_json(output / "progress.json", {
        "state": "running",
        "current_scenario": scenario["id"],
        "completed_scenarios": outcomes,
    })
    command = [
        sys.executable, "-u", str(ROOT / "experiments/run_sensitivity.py"),
        str(folder), "--split", "test",
        "--alpha", str(scenario["alpha"]),
        "--capacity-kw", str(scenario["capacity_kw"]),
        "--base-hours", str(scenario["base_hours"]),
        "--static-time-limit", str(scenario["static_time_limit"]),
        "--frozen-config-sha256", design["e1_config_sha256"],
        "--design", str(design_path),
        "--design-sha256", expected_hash,
        "--scenario-id", scenario["id"],
    ]
    write_json(folder / "launch.json", {
        "command": command,
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    outcome = {"scenario_id": scenario["id"], "status": "failed"}
    try:
        with (folder / "stdout_stderr.txt").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        outcome["returncode"] = completed.returncode
        write_json(folder / "process_exit.json", {
            "returncode": completed.returncode
        })
        if completed.returncode != 0:
            raise RuntimeError("Scenario process failed")
        verification = json.loads(
            (folder / "verification.json").read_text(encoding="utf-8")
        )
        assert verification["status"] == "passed"
        for rel, value in verification["source_and_input_sha256"].items():
            assert digest(ROOT / rel) == value, rel
        for rel, value in verification["output_artifact_sha256"].items():
            assert digest(folder / rel) == value, rel
        frame = pd.read_csv(folder / "comparison.csv")
        assert len(frame) == 8
        frame["evidence_directory"] = str(folder.relative_to(ROOT))
        frames.append(frame)
        outcome.update(
            status=("requires_review" if verification["requires_review"] else "passed"),
            check_count=verification["check_count"],
            wall_seconds=verification["wall_seconds"],
        )
    except Exception:
        outcome["error"] = traceback.format_exc()
    outcomes.append(outcome)
    write_json(folder / "scenario_outcome.json", outcome)
    write_json(output / "outcomes.json", outcomes)
    pd.concat(frames, ignore_index=True).to_csv(
        output / "sensitivity_results.csv", index=False
    )
    print(json.dumps(outcome, ensure_ascii=False), flush=True)

for rel, value in design["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == value, rel
assert digest(design_path) == expected_hash

all_passed = all(o["status"] == "passed" for o in outcomes)
result = {
    "status": "passed" if all_passed else "requires_review",
    "scope": "E2/E3 prescribed grid; E1 shared reference reused",
    "design_sha256": expected_hash,
    "scenarios": outcomes,
    "reference_directory": str(base.relative_to(ROOT)),
    "results_sha256": digest(output / "sensitivity_results.csv"),
}
write_json(output / "batch_verification.json", result)
write_json(output / "progress.json", {
    "state": "finished", "completed_scenarios": outcomes
})
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if all_passed else 1)
