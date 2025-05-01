import math
import random
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from pydub import AudioSegment
from pydub.playback import play
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
import ollama
from PyPDF2 import PdfReader
import warnings
warnings.filterwarnings("ignore", message=".*tensor.storage.*")

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
        "system_prompt": "You are a helpful, personable IT consultant who thinks step-by-step and uses sequential logic to provide solutions to the technological problems users present to you.",
        "temperature": 0.3,
        "max_tokens": 25,
        "n": 2
    },
    "2": {
        "name": "Cybersecurity Expert",
        "system_prompt": "You are an expert in cybersecurity who thinks step-by-step and uses sequential logic to provide advice on securing systems, detecting vulnerabilities, and maintaining data privacy.",
        "temperature": 0.6,
        "max_tokens": 75,
        "frequency_penalty": 0.5
    },
    "3": {
        "name": "Friendly Troubleshooting Assistant",
        "system_prompt": "You are a friendly troubleshooting assistant who thinks step-by-step and uses sequential logic to help users solve common IT issues such as slow internet, printer problems, and frozen computers.",
        "temperature": 0.9,
        "max_tokens": 50
    }
}

# Function to generate a random ticket number
def generate_ticket_number():
    return f"TICKET-{random.randint(100000, 999999)}"

# Function to calculate password entropy
def password_entropy(password):
    """
    Calculates password entropy based on character variety and length.
    """
    charset_sizes = {
        "lowercase": 26,
        "uppercase": 26,
        "digits": 10,
        "special": 32  # Common ASCII special characters
    }

    # Determine character variety in password
    charset_used = sum(charset_sizes[c] for c in charset_sizes if any(
        ch.islower() if c == "lowercase"
        else ch.isupper() if c == "uppercase"
        else ch.isdigit() if c == "digits"
        else ch in "!@#$%^&*()-_=+[]{}|;:',.<>?/`~" for ch in password))

    # Calculate entropy
    entropy = len(password) * math.log2(charset_used) if charset_used > 0 else 0

    # Classify password strength
    if entropy < 28:
        return "Weak: Your password is too easy to guess."
    elif entropy < 36:
        return "Moderate: Add more length or complexity to your password."
    else:
        return "Strong: Your password is secure!"

# Function to load and chunk the handbook.txt file for RAG
def load_and_chunk_document(file_path, chunk_size=500, chunk_overlap=50):
    """
    Loads a text file with UTF-8 encoding and chunks it into smaller sections for easier retrieval.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
            content = file.read()
    except UnicodeDecodeError as e:
        print(f"Error decoding file {file_path}: {e}")
        return []

    if not content.strip():
        print("Error: No content was found in the file. Ensure it contains readable text.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = [
        {"id": f"chunk_{i}", "text": chunk, "metadata": {"source": os.path.basename(file_path)}}
        for i, chunk in enumerate(text_splitter.split_text(content))
    ]
    return chunks


# Set up ChromaDB for RAG
def setup_chroma_db(chunks, collection_name="handbook_knowledge"):
    """
    Sets up a ChromaDB collection using the handbook chunks.
    """
    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )
    return collection

# Retrieve relevant context from ChromaDB
def retrieve_context(collection, query, n_results=3):
    """
    Retrieves relevant context for a query from the ChromaDB collection.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    return [doc for doc_list in results.get("documents", []) for doc in doc_list]

# Function to play audio using PyDub
def text_to_speech(text):
    """
    Converts text to speech and plays the audio.
    Handles sequence length to avoid runtime errors.
    """
    MAX_SEQUENCE_LENGTH = 600
    truncated_text = text[:MAX_SEQUENCE_LENGTH]

    # Process the truncated text
    inputs = processor(text=truncated_text, return_tensors="pt")
    try:
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
        audio_file = "response.wav"

        # Save the speech to file
        with open(audio_file, "wb") as f:
            f.write(speech.numpy().tobytes())

        # Play the audio using PyDub
        sound = AudioSegment.from_wav(audio_file)
        play(sound)

    except RuntimeError as e:
        print(f"Error during text-to-speech generation: {e}")

# Function to select an agent
def select_agent():
    print("Select a chat agent for your scenario:")
    for key, agent in agents.items():
        print(f"{key}. {agent['name']}")
    choice = input("Enter the number of your chosen agent: ")
    return agents.get(choice, agents["1"])

# Main chat loop
def main():
    handbook_path = "c:/Users/labadmin/McClellandGAME450Project/ITConsultant/handbook.txt"  # Use plain text file
    handbook_chunks = load_and_chunk_document(handbook_path)

    if not handbook_chunks:
        print("Error: No content was found in the handbook. Ensure it contains readable text.")
        return

    collection = setup_chroma_db(handbook_chunks)  # Set up handbook retrieval

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
            strength = password_entropy(password)
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