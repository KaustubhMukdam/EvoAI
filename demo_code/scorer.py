def score_response(response: str) -> float:
    length = len(response.split())
    if length < 10:
        return 0.2
    elif length < 50:
        return 0.5
    elif "EvoAI" in response or "self-evolving" in response:
        return 0.9
    else:
        return 0.7
