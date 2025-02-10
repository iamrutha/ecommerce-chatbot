import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time  # ✅ Added for "Thinking..." effect

# ✅ Load the Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load FAISS Index
faiss_index = faiss.read_index("models/faiss_index.bin")

# ✅ Load FAQs
with open("models/faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

# ✅ Function to Get Best Answer with Confidence Score
def get_best_answer(user_query):
    query_embedding = model.encode([user_query])
    distances, closest_index = faiss_index.search(query_embedding, 1)

    best_match_index = int(closest_index[0][0])  # Convert NumPy int to Python int
    confidence_score = distances[0][0]  # Lower is better

    if confidence_score > 0.5:
        return "I'm not sure about that. Can you rephrase?"

    return faqs[best_match_index]["answer"]  

# ✅ Streamlit UI
st.title("💬 AI-Powered Customer Support Chatbot")
st.write("Ask me anything about orders, shipping, returns, and discounts!")

# ✅ Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Clear Chat Button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []  # Clears chat history
    st.success("Chat cleared!")

# ✅ Chat Interface
user_input = st.text_input("Type your query here 👇")

if st.button("Ask Chatbot"):
    if user_input:
        with st.spinner("🤖 Thinking..."):  # ✅ Show "Thinking..." effect
            time.sleep(1)  # Simulate processing delay
            response = get_best_answer(user_input)

        # ✅ Append conversation to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Chatbot", response))

# ✅ Display Chat History
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"🧑 **{speaker}:** {message}")
    else:
        st.markdown(f"🤖 **{speaker}:** {message}")
