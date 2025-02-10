import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Load the Sentence Transformer Model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # A lightweight & fast model
print("Model loaded successfully!")

# ✅ Load FAQs from JSON
with open("data/faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)["faqs"]

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# ✅ Generate Embeddings for Questions
print("Generating embeddings for FAQ questions...")
question_embeddings = model.encode(questions, convert_to_numpy=True)

# ✅ Create FAISS Index
embedding_dimension = question_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embedding_dimension)  # L2 Distance (Euclidean)
faiss_index.add(question_embeddings)

# ✅ Save FAISS Index and FAQs
faiss.write_index(faiss_index, "models/faiss_index.bin")
with open("models/faqs.json", "w", encoding="utf-8") as f:
    json.dump(faqs, f, indent=4)

print("FAISS index created and saved successfully!")
