# import argparse
# from src.training.evolution_loop import EvolutionLoop

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--mode", choices=["local","cloud"], default="local")
#     ap.add_argument("--model_cfg", default="config/model_local.yaml")
#     ap.add_argument("--evo_cfg", default="config/evolution_local.yaml")
#     ap.add_argument("--gens", type=int, default=3)
#     args = ap.parse_args()

#     loop = EvolutionLoop(args.model_cfg, args.evo_cfg, mode=args.mode)
#     loop.run(max_generations=args.gens)


import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.evolution_loop import EvolutionLoop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local","cloud"], default="local")
    ap.add_argument("--model_cfg", default="config/model_local.yaml")
    ap.add_argument("--evo_cfg", default="config/evolution_local.yaml")
    ap.add_argument("--gens", type=int, default=3)
    args = ap.parse_args()

    print(f"Starting EvoAI in {args.mode} mode...")
    print(f"Model config: {args.model_cfg}")
    print(f"Evolution config: {args.evo_cfg}")
    print(f"Generations: {args.gens}")

    loop = EvolutionLoop(args.model_cfg, args.evo_cfg, mode=args.mode)
    loop.run(max_generations=args.gens)
