import json, yaml, random, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np

class EvolutionEngine:
    def __init__(self, cfg_path):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.archive, self.generation, self.genealogy = [], 0, {}
        self.hist = []
        self.archive_size = self.cfg["evolution"]["archive_size"]
        self.population_size = self.cfg["evolution"]["population_size"]
        self.mutation_rate = self.cfg["evolution"]["mutation_rate"]

    def _sig(self, obj):
        return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def add(self, agent):
        agent["signature"] = self._sig(agent)
        agent["generation"] = self.generation
        agent["timestamp"] = datetime.utcnow().isoformat()
        if agent["signature"] in [a["signature"] for a in self.archive]:
            return False
        self.archive.append(agent)
        if len(self.archive) > self.archive_size:
            self.archive.sort(key=lambda x: x.get("performance", 0), reverse=True)
            self.archive = self.archive[: self.archive_size]
        return True

    def select_parents(self, k=2):
        if len(self.archive) < 2:
            return random.choices(self.archive, k=k) if self.archive else []
        sel = []
        for _ in range(k):
            cand = random.sample(self.archive, min(3, len(self.archive)))
            scored = []
            for c in cand:
                perf = c.get("performance", 0.0)
                age = self.generation - c.get("generation", self.generation)
                scored.append((perf + 0.05 * age, c))
            sel.append(max(scored, key=lambda x: x[0])[1])
        return sel

    def crossover(self, p1, p2):
        feats = list(set(p1.get("features", []) + p2.get("features", [])))
        better = p1 if p1.get("performance", 0) >= p2.get("performance", 0) else p2
        return {"features": feats, "approach": better.get("approach", "standard")}

    def mutate(self, ag):
        mutations = ["add_logging", "improve_prompt", "error_handling", "multi_solution_rank", "input_validation"]
        if random.random() < self.mutation_rate:
            if "features" not in ag: ag["features"] = []
            ag["features"].append(random.choice(mutations))
        return ag

    def next_population(self):
        if not self.archive:
            return [{"agent_id": f"boot_{i}", "approach": "standard"} for i in range(self.population_size)]
        pop = []
        for i in range(self.population_size):
            if random.random() < 0.8:
                p = self.select_parents(2)
                child = self.crossover(p[0], p[1]) if len(p) == 2 else p[0].copy()
                child["agent_id"] = f"gen{self.generation}_offs_{i}"
            else:
                child = {"agent_id": f"gen{self.generation}_rand_{i}", "approach": "random"}
            pop.append(self.mutate(child))
        return pop

    def evolve(self, eval_results):
        self.generation += 1
        for r in eval_results:
            a = {
                "agent_id": f"g{self.generation}_{r['id']}",
                "performance": r["performance"],
                "constitutional_score": r.get("constitutional_score", 0.0),
                "features": r.get("features", []),
                "approach": r.get("approach", "standard"),
            }
            self.add(a)
        if self.archive:
            perfs = [a["performance"] for a in self.archive]
            self.hist.append({
                "generation": self.generation,
                "max": float(np.max(perfs)),
                "mean": float(np.mean(perfs)),
                "std": float(np.std(perfs)),
                "size": len(self.archive),
            })
