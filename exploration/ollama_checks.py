from ollama import chat

response = chat(
    model='llama3.2:1b',
    messages=[
        {'role': 'user', 'content': 'Hello!'}
    ]
)

print(response['message']['content'])
