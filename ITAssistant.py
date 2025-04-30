from ollama import chat

# Define the model and initial messages
model = 'llama3.2'
messages = [
    {
        'role': 'system',
        'content': (
            'You are a helpful, personable IT consultant who can think step by step to provide '
            'solutions to the technological problems the users present you with. '
            'You should be polite, engaging, and practical in your advice.'
        ),
    },
]
options = {
    'temperature': 0.3,  # Determines creativity of responses
    'max_tokens': 100,   # Increases response length for more detailed solutions
}

print("Welcome to the IT Help Desk Assistant. Type '/exit' to leave the session.")

# Conversation loop
while True:
    # Take user input
    user_input = input('You: ')
    message = {'role': 'user', 'content': user_input}
    messages.append(message)

    # Exit condition
    if user_input.strip().lower() == '/exit':
        print("Agent: Goodbye! Feel free to reach out anytime.")
        break

    # Call the chat function and append the assistant's response
    response = chat(model=model, messages=messages, stream=False, options=options)
    print(f'Agent: {response.message.content}')
    messages.append({'role': 'assistant', 'content': response.message.content})

    # Optional: Add custom commands or keywords for special functionality
    if "printer status" in user_input.lower():
        print("Agent: Checking the printer status... (Placeholder for RAG integration)")
        # Future integration: RAG for fetching printer data
    elif "troubleshoot" in user_input.lower():
        print("Agent: Let's go step by step. What issue are you experiencing?")
        # Future integration: Modular troubleshooting workflows
    elif "recommend upgrade" in user_input.lower():
        print("Agent: Let me analyze your system requirements... (Placeholder for system diagnostics)")
        # Future integration: AI-driven system upgrade recommendations
