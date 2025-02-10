import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# # ✅ Load the Embedding Model
# st.write("🔄 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
# st.success("✅ Model loaded successfully!")

# # ✅ Load FAISS Index
# st.write("🔄 Loading FAISS index...")
faiss_index = faiss.read_index("models/faiss_index.bin")
# st.success("✅ FAISS index loaded successfully!")

# ✅ Load FAQs
with open("data/faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

# ✅ Function to Get Best Answer with Confidence Score
def get_best_answer(user_query):
    query_embedding = model.encode([user_query])
    distances, closest_index = faiss_index.search(query_embedding, 1)

    best_match_index = closest_index[0][0]
    confidence_score = distances[0][0]  # Lower is better

    if confidence_score > 0.5:
        return "I'm not sure about that. Can you rephrase?"

    return faqs[best_match_index]["answer"]

# ✅ Streamlit UI
st.title("💬 AI-Powered Customer Support Chatbot")
st.write("Ask me anything about orders, shipping, returns, and discounts!")

# Chat Interface
user_input = st.text_input("Type your query here 👇")

if st.button("Ask Chatbot"):
    if user_input:
        response = get_best_answer(user_input)
        st.markdown(f"**Chatbot:** {response}")
