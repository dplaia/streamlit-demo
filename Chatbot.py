from openai import OpenAI
from google import genai
from google.genai import types
import streamlit as st

with st.sidebar:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    "Upload an image"

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot Demo")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # Read OpenAI API key from system environment variable
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Only run this block for Google AI API
    client = genai.Client(api_key=GEMINI_API_KEY)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in client.models.generate_content_stream(model='gemini-2.0-flash-exp', contents=prompt):
            if chunk.text:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
