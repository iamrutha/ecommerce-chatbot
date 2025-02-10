import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load the Embedding Model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# ✅ Load FAISS Index
print("Loading FAISS index...")
faiss_index = faiss.read_index("models/faiss_index.bin")

# ✅ Load FAQs
with open("models/faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

# ✅ Function to Get Best Answer with Confidence Score
def get_best_answer(user_query):
    query_embedding = model.encode([user_query])
    distances, closest_index = faiss_index.search(query_embedding, 1)  # Find closest match

    best_match_index = closest_index[0][0]
    confidence_score = distances[0][0]  # Lower is better (L2 Distance)

    print(f"DEBUG: Best match index = {best_match_index}, Confidence = {confidence_score:.4f}")

    # ✅ Set Confidence Threshold (Lower is better)
    if confidence_score > 0.5:  # Adjust threshold if needed
        return "I'm not sure about that. Can you rephrase?"

    return faqs[best_match_index]["answer"]

# ✅ Run Chatbot
print("Chatbot Ready! Type your query or type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = get_best_answer(user_input)
    print(f"Chatbot: {response}")
