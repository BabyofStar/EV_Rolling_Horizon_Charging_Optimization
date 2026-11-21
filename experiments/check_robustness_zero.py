
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
config_path, config_hash, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )

assert digest(config_path) == config_hash
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
for rel, expected in config["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, rel

for variant in ["bypass", "zero"]:
    folder = OUT / variant
    folder.mkdir(exist_ok=False)
    write_json(OUT / "progress.json", {"variant": variant, "state": "running"})
    command = [
        sys.executable, "-u", str(ROOT / "experiments/run_robustness.py"),
        str(folder), "--split", "test",
        "--frozen-config-sha256", config["e1_config_sha256"],
        "--protocol-config", str(config_path),
        "--protocol-sha256", config_hash,
        "--scenario-id", "zero_equivalence", "--smoke",
    ]
    if variant == "bypass":
        command.append("--bypass-transform")
    write_json(folder / "launch.json", {"command": command})
    with (folder / "stdout_stderr.txt").open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
    write_json(folder / "process_exit.json", {"returncode": process.returncode})
    assert process.returncode == 0, f"{variant}: process failed"
    verification = json.loads(
        (folder / "verification.json").read_text(encoding="utf-8")
    )
    assert verification["status"] == "passed"
    assert verification["requires_review"] is False
    for rel, expected in verification["source_and_input_sha256"].items():
        assert digest(ROOT / rel) == expected, rel
    for rel, expected in verification["output_artifact_sha256"].items():
        assert digest(folder / rel) == expected, rel
    print(f"{variant}: completed, six combinations audited", flush=True)

checks = []
for cohort in ["main_at_arrival", "supplement_after_submission"]:
    for method in ["Uncontrolled", "TOU_Greedy", "RH_LP"]:
        left = OUT / "bypass" / cohort / method
        right = OUT / "zero" / cohort / method
        for name in [
            "station_trajectory.csv", "session_trajectory.csv",
            "session_service.csv", "daily_metrics.csv"
        ]:
            a = pd.read_csv(left / name, dtype={"sessionID": str})
            b = pd.read_csv(right / name, dtype={"sessionID": str})
            pd.testing.assert_frame_equal(
                a, b, check_dtype=False, check_exact=False,
                rtol=0, atol=1e-6,
            )
            checks.append(f"{cohort}/{method}/{name}")

    # 求解时间预算的微小墙钟差异不属于模型输入差异
    aroot = OUT / "bypass" / cohort / "RH_LP/rh_decisions"
    broot = OUT / "zero" / cohort / "RH_LP/rh_decisions"
    afiles = {str(p.relative_to(aroot)): p for p in aroot.glob("*/case.json")}
    bfiles = {str(p.relative_to(broot)): p for p in broot.glob("*/case.json")}
    assert afiles and afiles.keys() == bfiles.keys()
    for key in afiles:
        a = json.loads(afiles[key].read_text(encoding="utf-8"))
        b = json.loads(bfiles[key].read_text(encoding="utf-8"))
        a.pop("time_limit_seconds", None)
        b.pop("time_limit_seconds", None)
        assert a == b, f"LP input mismatch: {cohort}/{key}"
    checks.append(f"{cohort}: identical LP cases except wall-clock budget")

for rel, expected in config["source_and_input_sha256"].items():
    assert digest(ROOT / rel) == expected, rel
assert digest(config_path) == config_hash

result = {
    "status": "passed",
    "scope": "E4 runner small-sample zero-error equivalence only",
    "check_count": len(checks),
    "checks": checks,
    "protocol_sha256": config_hash,
    "long_experiments_started": False,
    "run_verification_sha256": {
        variant: digest(OUT / variant / "verification.json")
        for variant in ["bypass", "zero"]
    },
}
write_json(OUT / "smoke_verification.json", result)
write_json(OUT / "progress.json", {"state": "finished"})
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
