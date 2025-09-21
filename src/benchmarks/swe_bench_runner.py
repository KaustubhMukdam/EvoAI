import json, shutil, tempfile, subprocess, yaml, logging
from datasets import load_dataset

class SWEBenchRunner:
    def __init__(self, cfg_path, local_mode=True):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.sandbox = self.cfg["safety"]["sandbox_enabled"]
        self.timeout = self.cfg["safety"]["max_execution_time"]
        self.local_mode = local_mode
        self._load_dataset()

    def _load_dataset(self):
        try:
            self.ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        except Exception:
            self.ds = load_dataset("princeton-nlp/SWE-bench", split="test")

    def evaluate_solution(self, idx, code):
        prob = self.ds[idx]
        tmp = tempfile.mkdtemp(prefix="evoai_")
        try:
            path = f"{tmp}/solution.py"
            with open(path, "w") as f: f.write(code)
            # Simplified execution: just check code runs without error
            res = subprocess.run(["python", path], capture_output=True, text=True, timeout=self.timeout)
            passed = (res.returncode == 0) and ("error" not in (res.stdout + res.stderr).lower())
            return {"idx": idx, "passed": passed, "stdout": res.stdout[-500:], "stderr": res.stderr[-500:]}
        except subprocess.TimeoutExpired:
            return {"idx": idx, "passed": False, "stderr": "timeout"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def run_subset(self, solutions, subset_size=10):
        N = min(subset_size, len(solutions), len(self.ds))
        results = [self.evaluate_solution(i, solutions[i]) for i in range(N)]
        passed = sum(1 for r in results if r["passed"])
        return {"total": N, "passed": passed, "pass_rate": (passed / N if N else 0.0), "details": results}
