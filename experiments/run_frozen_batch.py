
"""Run archived frozen batches; does not regenerate or retune protocols."""
from pathlib import Path
import argparse
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["sensitivity", "robustness"])
parser.add_argument("--output", required=True)
args = parser.parse_args()
choices = {
    "sensitivity": (
        "configs/sensitivity.yaml",
        "7079f7033ae1bfd38d2bbf4a76115cfc1fe002137b779d034b3c4af36d1f23e4",
        "experiments/run_sensitivity_batch.py"),
    "robustness": (
        "configs/robustness.yaml",
        "5c52d926f1f94598f9f285babd0cacf0365db4b7fc085029f70d7be2618aa5e8",
        "experiments/run_robustness_batch.py"),
}
config, expected, entry = choices[args.kind]
config = ROOT / config
with config.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
if actual != expected:
    raise RuntimeError("Frozen configuration fingerprint mismatch")
out = Path(args.output)
if not out.is_absolute():
    out = ROOT / out
out = out.resolve()
if not out.is_relative_to(ROOT):
    raise ValueError("Output must be inside this project")
out.mkdir(parents=True, exist_ok=False)
completed = subprocess.run(
    [sys.executable, str(ROOT / entry), str(config), expected, str(out)],
    cwd=ROOT)
sys.exit(completed.returncode)
