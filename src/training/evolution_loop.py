# import os, json, yaml, logging, time
# from pathlib import Path
# from .utils import ensure_dir  # optional: inline ensure_dir if not using utils
# from ..models.qwen_base import QwenBaseModel
# from ..models.evolution_engine import EvolutionEngine
# from ..benchmarks.swe_bench_runner import SWEBenchRunner

# def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# class EvolutionLoop:
#     def __init__(self, model_cfg, evo_cfg, mode="local"):
#         self.model = QwenBaseModel(model_cfg)
#         self.engine = EvolutionEngine(evo_cfg)
#         self.runner = SWEBenchRunner(evo_cfg, local_mode=(mode=="local"))
#         with open(evo_cfg, "r") as f:
#             self.ecfg = yaml.safe_load(f)
#         ensure_dir("experiments/logs"); ensure_dir("experiments/results"); ensure_dir("models/evolved_models")
#         logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#         self.mode = mode
#         self.subset_sizes = self.ecfg["evaluation"]["subset_sizes"]

#     def _solutions_for_agent(self, agent, upto=10):
#         sols = []
#         for i in range(upto):
#             prob = self.runner.ds[i]["problem_statement"] if "problem_statement" in self.runner.ds.features else str(self.runner.ds[i])
#             if agent.get("approach") in ("constitutional_enhanced",) or ("improve_prompt" in agent.get("features", [])):
#                 data = self.model.constitutional_generation(prob, max_revisions=2 if self.mode=="local" else 3)
#                 code = data["final_code"]
#             else:
#                 code = self.model.generate_code(prob)
#             sols.append(code)
#         return sols

#     def evaluate_agent(self, agent):
#         # staged subsets
#         stage_best = 0.0
#         last = None
#         sols = self._solutions_for_agent(agent, upto=max(self.subset_sizes))
#         for sz in self.subset_sizes:
#             res = self.runner.run_subset(sols, subset_size=sz)
#             stage_best = res["pass_rate"]
#             last = res
#             if stage_best < self.ecfg["evaluation"]["promotion_threshold"]:
#                 break
#         const_scores = []
#         for i, code in enumerate(sols[:min(5, len(sols))]):
#             prob = self.runner.ds[i]["problem_statement"] if "problem_statement" in self.runner.ds.features else str(self.runner.ds[i])
#             const_scores.append(self.model.score_constitution(prob, code))
#         avg_const = sum(const_scores)/len(const_scores) if const_scores else 0.0
#         alpha = self.ecfg["rewards"]["alpha_tests_passed"]; beta = self.ecfg["rewards"]["beta_constitution_score"]
#         combined = alpha * stage_best + beta * avg_const
#         return {"id": agent["agent_id"], "performance": stage_best, "constitutional_score": avg_const, "combined_reward": combined, "stage": last}

#     def run_generation(self, gen_idx):
#         pop = self.engine.next_population()
#         evals = [self.evaluate_agent(a) for a in pop]
#         self.engine.evolve(evals)
#         # persist results
#         out = {"generation": gen_idx, "results": evals}
#         with open(f"experiments/results/gen_{gen_idx:03d}.json", "w") as f: json.dump(out, f, indent=2)
#         return evals

#     def run(self, max_generations=3):
#         for g in range(1, max_generations+1):
#             self.run_generation(g)


import os, json, yaml, logging, time
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen_base import QwenBaseModel
from models.evolution_engine import EvolutionEngine
from benchmarks.swe_bench_runner import SWEBenchRunner

def ensure_dir(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

class EvolutionLoop:
    def __init__(self, model_cfg, evo_cfg, mode="local"):
        self.model = QwenBaseModel(model_cfg)
        self.engine = EvolutionEngine(evo_cfg)
        self.runner = SWEBenchRunner(evo_cfg, local_mode=(mode=="local"))
        with open(evo_cfg, "r") as f:
            self.ecfg = yaml.safe_load(f)
        ensure_dir("experiments/logs")
        ensure_dir("experiments/results") 
        ensure_dir("models/evolved_models")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("experiments/logs/evolution.log"),
                logging.StreamHandler()
            ]
        )
        
        self.mode = mode
        self.subset_sizes = self.ecfg["evaluation"]["subset_sizes"]
        logging.info(f"EvolutionLoop initialized in {mode} mode")

    def _solutions_for_agent(self, agent, upto=10):
        sols = []
        logging.info(f"Generating {upto} solutions for agent {agent.get('agent_id', 'unknown')}")
        
        for i in range(min(upto, len(self.runner.ds))):
            # Extract problem statement safely
            prob_data = self.runner.ds[i]
            if isinstance(prob_data, dict) and "problem_statement" in prob_data:
                prob = prob_data["problem_statement"]
            else:
                prob = f"Solve programming problem {i}: {str(prob_data)[:200]}"
            
            # Generate solution based on agent approach
            if agent.get("approach") == "constitutional_enhanced" or "improve_prompt" in agent.get("features", []):
                data = self.model.constitutional_generation(prob, max_revisions=2 if self.mode=="local" else 3)
                code = data["final_code"]
            else:
                code = self.model.generate_code(prob)
            sols.append(code)
            
        logging.info(f"Generated {len(sols)} solutions")
        return sols

    def evaluate_agent(self, agent):
        logging.info(f"Evaluating agent: {agent.get('agent_id', 'unknown')}")
        
        # Generate solutions up to max subset size
        max_size = max(self.subset_sizes) if self.subset_sizes else 10
        sols = self._solutions_for_agent(agent, upto=max_size)
        
        # Run staged evaluation
        stage_best = 0.0
        last_result = None
        
        for sz in self.subset_sizes:
            logging.info(f"Running subset evaluation with {sz} problems")
            res = self.runner.run_subset(sols, subset_size=sz)
            stage_best = res["pass_rate"]
            last_result = res
            
            logging.info(f"Stage {sz}: Pass rate = {stage_best:.2%}")
            
            # Check promotion threshold
            if stage_best < self.ecfg["evaluation"]["promotion_threshold"]:
                logging.info(f"Failed to meet promotion threshold {self.ecfg['evaluation']['promotion_threshold']:.2%}")
                break
        
        # Calculate constitutional score on sample
        const_scores = []
        sample_size = min(5, len(sols))
        for i in range(sample_size):
            prob_data = self.runner.ds[i]
            if isinstance(prob_data, dict) and "problem_statement" in prob_data:
                prob = prob_data["problem_statement"]
            else:
                prob = f"Programming problem {i}"
            const_scores.append(self.model.score_constitution(prob, sols[i]))
        
        avg_const = sum(const_scores)/len(const_scores) if const_scores else 0.0
        
        # Calculate combined reward
        alpha = self.ecfg["rewards"]["alpha_tests_passed"]
        beta = self.ecfg["rewards"]["beta_constitution_score"]
        combined = alpha * stage_best + beta * avg_const
        
        result = {
            "id": agent["agent_id"], 
            "performance": stage_best, 
            "constitutional_score": avg_const, 
            "combined_reward": combined, 
            "stage_result": last_result
        }
        
        logging.info(f"Agent {agent['agent_id']}: Performance={stage_best:.3f}, Constitutional={avg_const:.3f}")
        return result

    def run_generation(self, gen_idx):
        logging.info(f"\n{'='*50}")
        logging.info(f"GENERATION {gen_idx}")
        logging.info(f"{'='*50}")
        
        # Get population for this generation
        pop = self.engine.next_population()
        logging.info(f"Population size: {len(pop)}")
        
        # Evaluate each agent
        evals = []
        for i, agent in enumerate(pop):
            logging.info(f"Evaluating agent {i+1}/{len(pop)}")
            result = self.evaluate_agent(agent)
            evals.append(result)
        
        # Evolve to next generation
        self.engine.evolve(evals)
        
        # Save results
        result_data = {
            "generation": gen_idx,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(pop),
            "results": evals,
            "best_performance": max(r["performance"] for r in evals) if evals else 0.0
        }
        
        result_file = f"experiments/results/gen_{gen_idx:03d}.json"
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        
        logging.info(f"Generation {gen_idx} complete. Results saved to {result_file}")
        return evals

    def run(self, max_generations=3):
        logging.info(f"Starting evolution run: {max_generations} generations in {self.mode} mode")
        start_time = time.time()
        
        try:
            for g in range(1, max_generations + 1):
                gen_start = time.time()
                self.run_generation(g)
                gen_time = time.time() - gen_start
                logging.info(f"Generation {g} completed in {gen_time:.1f} seconds")
                
        except KeyboardInterrupt:
            logging.info("Evolution interrupted by user")
        except Exception as e:
            logging.error(f"Error during evolution: {e}", exc_info=True)
        
        total_time = time.time() - start_time
        logging.info(f"Evolution run complete. Total time: {total_time:.1f} seconds")
