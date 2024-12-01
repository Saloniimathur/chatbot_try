from dataclasses import dataclass
import streamlit as st
#from langchain_community.llms import CTransformers
from textblob import TextBlob
import os
from transformers import pipeline
from huggingface_hub.inference_api import InferenceApi

@dataclass
class Message:
    actor: str
    payload: str

# Define the actor roles
USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"

# Initialize chat history in session state
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [
        Message(actor=ASSISTANT, payload="Hi! I'm here to help you with your math questions. Please enter your question and how you're feeling about it.")
    ]
    st.session_state["conversation_step"] = 1  # Track the conversation step

# Function to load API key from a text file
def load_api_key(filepath: str):
    try:
        with open(filepath, 'r') as file:
            return file.read().strip()
    except Exception as e:
        st.error(f"Error reading API key from file: {e}")
        return None
 
# Load the API key
api_key = load_api_key('hugging_face_token.txt')
if not api_key:
    st.error("Failed to load API key from file.")
    st.stop()

# Initialize the Inference API with the model and API key
llm = InferenceApi("TheBloke/Llama-2-7B-Chat-GGML", token=api_key)


# Display existing chat messages
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

# Accept user input for a multi-turn conversation
user_input = st.text_input("Enter your response here:")

if user_input:
    # Analyze the sentiment of the user's input for emotion context
    emotion_analysis = TextBlob(user_input)
    sentiment_score = emotion_analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

    # Add user input to session state and display it
    st.session_state[MESSAGES].append(Message(actor=USER, payload=user_input))
    st.chat_message(USER).write(user_input)

    # Load the model
    llm = InferenceApi("TheBloke/Llama-2-7B-Chat-GGML", token=api_key)
    if llm:
        try:
            if st.session_state["conversation_step"] == 1:
                # Initial encouragement and question to start the problem-solving process
                emotion_context = (
                    "You seem a bit anxious, but don't worry, I'm here to guide you!" if sentiment_score < -0.1 else
                    "You seem confident! Let's tackle this together." if sentiment_score > 0.1 else
                    "You're feeling neutral, let's take it step by step."
                )
                response_prompt = (
                    f"{emotion_context} Can you tell me what you think the first step should be for solving this problem?"
                )
                response = response_prompt
                st.session_state["conversation_step"] += 1 

            else:
                # Continue guiding the user based on their input
                response_prompt = (
                    f"You mentioned: '{user_input}'. Let's continue from there. What do you think should be the next step?"
                )
                response = llm(response_prompt)

            # Add and display the assistant's response
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
            st.chat_message(ASSISTANT).write(response)
        except Exception as e:
            st.error(f"Error generating response: {e}")