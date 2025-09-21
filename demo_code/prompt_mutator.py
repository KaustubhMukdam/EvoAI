import random

def mutate_prompt(prompt: str) -> str:
    mutations = [
        "Explain it like I'm a beginner.",
        "Give a more detailed and technical explanation.",
        "Make it concise and focused on benefits.",
        "How would a researcher define it?",
    ]
    return prompt + " " + random.choice(mutations)
