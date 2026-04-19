"""
Runs every training script in its own subprocess (fresh Python process = full
memory wipe between models) then generates the comparison plots.
"""
import subprocess
import sys
import time
import os

SCRIPTS = [
    'train_isolation_forest.py',
    'train_lof.py',
    'train_ocsvm.py',
    'train_knn.py',
    'train_xgboost.py',
    'train_autoencoder.py',
]

failed = []

for script in SCRIPTS:
    print(f"\n{'='*60}", flush=True)
    print(f"  {script}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) — skipping.", flush=True)
        failed.append(script)
    else:
        print(f"\n  Done in {elapsed:.1f}s", flush=True)

print(f"\n{'='*60}", flush=True)
if failed:
    print(f"  Skipped (failed): {', '.join(failed)}", flush=True)

print("  Running compare.py ...", flush=True)
print(f"{'='*60}", flush=True)
subprocess.run([sys.executable, 'compare.py'],
               cwd=os.path.dirname(os.path.abspath(__file__)))

print("\nAll done.", flush=True)
