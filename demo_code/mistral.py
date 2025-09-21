import ollama

response = ollama.chat(model='mistral', messages=[
  {'role': 'user', 'content': 'Explain EvoAI briefly.'}
])

print(response['message']['content'])
