"""Run every check, write one log per script to results/, and a summary."""

import datetime
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = sorted(f for f in os.listdir(os.path.join(HERE, "checks")) if f.startswith("check_") and f.endswith(".py"))


def versions():
    out = [f"python {platform.python_version()}"]
    for mod in ("flint", "sympy", "numpy"):
        try:
            m = __import__(mod)
            out.append(f"{'python-flint' if mod == 'flint' else mod} {m.__version__}")
        except Exception:
            out.append(f"{mod} missing")
    return ", ".join(out)


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    only = sys.argv[1:]
    lines = [f"run: {datetime.datetime.now().isoformat(timespec='seconds')}", versions(), ""]
    failed = []
    for name in CHECKS:
        if only and not any(o in name for o in only):
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, os.path.join(HERE, "checks", name)],
                              capture_output=True, text=True)
        dt = time.time() - t0
        log = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
        with open(os.path.join(HERE, "results", name.replace(".py", ".log")), "w") as fh:
            fh.write(log)
        tail = [l for l in proc.stdout.splitlines() if l.strip().startswith("--")]
        status = "PASS" if proc.returncode == 0 else "FAIL"
        if proc.returncode:
            failed.append(name)
        line = f"{status}  {name:34s} {tail[-1].strip() if tail else ''}  ({dt:.0f}s)"
        print(line, flush=True)
        lines.append(line)
    lines.append("")
    lines.append("ALL PASSED" if not failed else "FAILED: " + ", ".join(failed))
    print(lines[-1])
    if not only:
        with open(os.path.join(HERE, "results", "SUMMARY.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
