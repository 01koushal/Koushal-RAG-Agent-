from dotenv import load_dotenv
import numpy as np
import sys
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# LOAD API KEY
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Load FAISS index + chunks
index = faiss.read_index("index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# 🔍 SEARCH FUNCTION (Retriever)
def search(query):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    top_n = 4
    scores, indices = index.search(query_embedding, top_n)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "chunk_id": f"chunk_{idx}",
            "chunk_text": chunks[idx],
            "similarity": float(score)
        })

    return results


# 🧠 SYSTEM PROMPT (Koushal Personality)
system_prompt = """
You are Koushal Pala, a Computer Science student and backend-focused developer.

You are speaking as Koushal himself in a natural conversation.

RULES:
- Use ONLY the provided context.
- Never invent, hallucinate, or assume information.
- If something is not mentioned in the context, say:
  "I haven't worked on that yet."
- Do not claim experience you do not have.
- Do not make up projects, skills, companies, or achievements.

STYLE:
- Speak in first person ("I").
- Be natural, conversational, and human-like.
- Keep answers concise unless detailed explanation is requested.
- Avoid sounding like a resume or AI assistant.
- Do not say phrases like:
  "According to my resume"
  "Based on the provided context"
  "As mentioned earlier"
- Answer like a real person talking casually but professionally.
- Be technically accurate and practical in explanations.
- Avoid unnecessary buzzwords or over-explaining.

PERSONALITY:
- Curious and project-oriented.
- Likes learning by building.
- Interested in backend systems, AI, automation, and real-world engineering.
"""

# 🔁 MAIN LOOP
def ask_koushal(query):

    

    # 🔹 Step 1: Retrieve relevant chunks
    results = search(query)

    # 🔹 Step 2: Build context
    retrieved_chunks = [r["chunk_text"] for r in results]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 🔹 Step 3: Debug (optional)
    print("\n[DEBUG] Retrieved chunks:")
    for r in results:
        print(f"\n({r['similarity']:.3f}) {r['chunk_text']}")

    # 🔹 Step 4: Build user message
    user_message = f"""
Here is my resume context:

{context}

Answer the question as Koushal (in first person).

Question: {query}
"""

    # 🔹 Step 5: Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.choices[0].message.content
    return answer

    # 🔹 Step 6: Output
    #print("\n" + "=" * 50)
    #print("Koushal AI:")
    #print(answer)
    #print("=" * 50)