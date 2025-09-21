import ollama
from demo_code.scorer import score_response
from demo_code.prompt_mutator import mutate_prompt

MODELS = ['mistral', 'llama3']
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.75


def get_response(model_name, prompt):
    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        return f"[{model_name} Error] {e}"


def evoai_main(prompt):
    print(f"\nStarting EvoAI with prompt: {prompt}")
    iteration = 0
    best_response = None
    best_score = 0.0

    while iteration < MAX_ITERATIONS:
        print(f"\nIteration {iteration + 1}")
        responses = {}

        for model in MODELS:
            print(f"Asking {model}...")
            answer = get_response(model, prompt)
            print(f"📥 {model} response:\n{answer}\n")
            responses[model] = answer

        for model, answer in responses.items():
            score = score_response(answer)
            print(f"{model} score: {score:.2f}")
            if score > best_score:
                best_score = score
                best_response = answer

        if best_score >= CONFIDENCE_THRESHOLD:
            print("\nConfidence threshold met. Stopping early.")
            break

        # If not confident, mutate prompt and retry
        prompt = mutate_prompt(prompt)
        print(f"\n🧬 Prompt mutated to: {prompt}")
        iteration += 1

    print("\n🎯 Final selected answer:")
    print(best_response)
    return best_response


if __name__ == "__main__":
    choice = input("1. Enter question\n2. Load from file\nChoose: ")

    if choice == "2":
        file_path = input("Enter path to code file: ")
        with open(file_path, 'r') as f:
            code = f.read()
        user_prompt = f"""
            Act as an expert Python developer. Analyze the following code for:
            - Syntax errors
            - Logical errors
            - Code quality improvements

            Fix the issues and return only the corrected code.

            ```python
            {code}
        """
    else:
        user_prompt = input("Enter your question: ")

    evoai_main(user_prompt)
