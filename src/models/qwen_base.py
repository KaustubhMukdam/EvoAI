import os
import random
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenBaseModel:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.model_name = self.cfg["model"]["name"]
        self.device = self.cfg["model"]["device"]
        self.max_length = self.cfg["model"]["max_length"]
        self.temperature = self.cfg["model"]["temperature"]
        self.top_p = self.cfg["model"]["top_p"]
        self.cache_dir = self.cfg["model"].get("cache_dir", None)
        self._load_model()
        self._setup_constitution()

    def _load_model(self):
        quant = self.cfg["model"].get("quantization", "none")
        kwargs = dict(cache_dir=self.cache_dir, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cuda" and quant == "4bit":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.float16,
                **kwargs
            )
        else:
            # CPU path with int8 quantization handled by torch dynamic quant where applicable
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                **kwargs
            ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_constitution(self):
        self.principles = [
            "Ensure security; avoid unsafe operations or injections.",
            "Prefer readable, documented code with clear naming.",
            "Handle errors and edge cases explicitly.",
            "Never access external resources without explicit tools.",
            "Be efficient; avoid unnecessary complexity.",
            "Preserve public APIs and interfaces.",
            "Include simple tests or validation hints when possible.",
            "Follow least-privilege and minimal scope.",
            "Keep changes maintainable and modular.",
            "Preserve original intent while improving functionality.",
        ]

    def _generate(self, prompt, max_new_tokens=512):
        toks = self.tokenizer(prompt, return_tensors="pt")
        toks = {k: v.to(self.model.device) for k, v in toks.items()}
        with torch.no_grad():
            out = self.model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][toks["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate_code(self, problem, max_new_tokens=256):
        prompt = f"Solve this programming problem briefly and safely:\n{problem}\n\nSolution:\n"
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def critique_code(self, problem, code, principle):
        prompt = f"""Review the proposed code against the principle: "{principle}".
Problem:\n{problem}\n\nCode:\n{code}\n\nList concrete issues or say 'complies' if acceptable:\n"""
        return self._generate(prompt, max_new_tokens=192)

    def revise_code(self, code, critique, principle):
        prompt = f"""Revise the code to address the critique while keeping behavior.
Principle: "{principle}"
Critique:\n{critique}\n\nOriginal Code:\n{code}\n\nRevised Code:\n"""
        return self._generate(prompt, max_new_tokens=256)

    def constitutional_generation(self, problem, max_revisions=2):
        code = self.generate_code(problem)
        history = []
        for _ in range(max_revisions):
            principle = random.choice(self.principles)
            crit = self.critique_code(problem, code, principle)
            if any(w in crit.lower() for w in ["complies", "looks good", "no issues"]):
                break
            new_code = self.revise_code(code, crit, principle)
            history.append({"principle": principle, "critique": crit, "code": new_code})
            code = new_code
        return {"initial_code": history[0]["code"] if history else code, "final_code": code, "revision_history": history}

    def score_constitution(self, problem, code):
        pos, neg = 0, 0
        for p in self.principles:
            crit = self.critique_code(problem, code, p)
            if any(k in crit.lower() for k in ["violation", "issue", "problem", "unsafe"]):
                neg += 1
            elif any(k in crit.lower() for k in ["comply", "good", "appropriate", "secure"]):
                pos += 1
        total = len(self.principles)
        return max(0.0, (pos - neg) / total)
