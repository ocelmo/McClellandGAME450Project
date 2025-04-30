#!/usr/bin/env python3
import random
import ollama
import chromadb
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from pydub import AudioSegment
from pydub.playback import play
from passlib.pwd import strength


# Initialize the SpeechT5 model and processor for text-to-speech
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)

# Define chatbot agents with different parameters
agents = {
    "1": {
        "name": "General IT Consultant",
        "system_prompt": "You are a helpful, personable IT consultant who thinks step by step to provide solutions to the technological problems users present to you.",
        "temperature": 0.3,
        "max_tokens": 25,
        "n": 2
    },
    "2": {
        "name": "Cybersecurity Expert",
        "system_prompt": "You are an expert in cybersecurity, focused on providing advice on securing systems, detecting vulnerabilities, and maintaining data privacy.",
        "temperature": 0.6,
        "max_tokens": 75,
        "frequency_penalty": 0.5
    },
    "3": {
        "name": "Friendly Troubleshooting Assistant",
        "system_prompt": "You are a friendly troubleshooting assistant who uses relatable language and humor to help users solve common IT issues such as slow internet, printer problems, and frozen computers.",
        "temperature": 0.9,
        "max_tokens": 50,
        "context_window": 8192  # Extended memory for chat continuity
    }
}

# Function to generate a random ticket number
def generate_ticket_number():
    return f"TICKET-{random.randint(100000, 999999)}"

# Function to check password strength
def check_password_strength(password):
    entropy = strength(password)
    if entropy < 28:
        return "Weak: Your password is too easy to guess."
    elif entropy < 36:
        return "Moderate: Add more length or complexity to your password."
    else:
        return "Strong: Your password is secure!"

# Function to retrieve IT handbook data using ChromaDB
def setup_chroma_db():
    client = chromadb.Client()
    try:
        client.delete_collection("handbook_knowledge")
    except:
        pass
    collection = client.create_collection(name="handbook_knowledge")
    return collection

def retrieve_context(collection, query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return [doc for doc_list in results.get("documents", []) for doc in doc_list]

# Function to convert text to speech
def text_to_speech(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    audio_file = "response.wav"
    
    # Save generated speech to file
    with open(audio_file, "wb") as f:
        f.write(speech.numpy().tobytes())

    # Play the audio using PyDub
    sound = AudioSegment.from_wav(audio_file)
    play(sound)

# Function to select an agent
def select_agent():
    print("Select a chat agent for your scenario:")
    for key, agent in agents.items():
        print(f"{key}. {agent['name']}")
    choice = input("Enter the number of your chosen agent: ")
    return agents.get(choice, agents["1"])

# Main chat loop
def main():
    collection = setup_chroma_db()  # Set up handbook retrieval

    selected_agent = select_agent()
    model = 'llama3.2'
    messages = [{'role': 'system', 'content': selected_agent['system_prompt']}]
    options = {
        'temperature': selected_agent['temperature'],
        'max_tokens': selected_agent['max_tokens']
    }

    print(f"You are now chatting with: {selected_agent['name']}. Type '/exit' to leave.")

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() == '/exit':
            print("Agent: Goodbye!")
            break
        elif "ticket number" in user_input.lower():
            ticket_number = generate_ticket_number()
            print(f"Agent: Your ticket number is {ticket_number}.")
            text_to_speech(f"Your ticket number is {ticket_number}.")
            continue
        elif "password strength" in user_input.lower():
            password = input("Enter your password for evaluation: ")
            strength = check_password_strength(password)
            print(f"Agent: {strength}")
            text_to_speech(strength)
            continue
        elif "consultant handbook" in user_input.lower():
            context = retrieve_context(collection, user_input)
            context_text = "\n\n".join(context) if context else "No relevant information found in the handbook."
            user_input += f"\n\nContext: {context_text}"

        messages.append({'role': 'user', 'content': user_input})
        response = ollama.chat(model=model, messages=messages, stream=False, options=options)
        print(f"Agent: {response.message.content}")
        text_to_speech(response.message.content)
        messages.append({'role': 'assistant', 'content': response.message.content})

if __name__ == "__main__":
    main()
