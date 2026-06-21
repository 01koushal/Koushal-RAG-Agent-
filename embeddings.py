
         
import pickle
import numpy as np                       
import faiss                              # FAISS: Facebook AI Similarity Search — production vector search.
#from sentence_transformers import SentenceTransformer  # Converts text into semantic embeddings.      
from google import genai
from dotenv import load_dotenv
import os   
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(
    api_key=gemini_key
)





doc_path = "document.txt"  # Path to the text file containing documents to index.
print(f"Loading documents from {doc_path}...")

with open(doc_path,"r") as f:
    documents = f.read()  
print(f"Loaded {len(documents)} characters.")



# Splitting documents into chunks
def split_into_chunks(text, chunk_size=400, overlap=100):
    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap  

    return chunks

chunks = split_into_chunks(documents, chunk_size=400, overlap=100)
print(f"Split into {len(chunks)} chunks.")



#Create Embeddings

print("\nLoading embedding model (first run downloads ~30MB model)...")

chunk_embeddings = []

for chunk in chunks:
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk
    )

    chunk_embeddings.append(
        response.embeddings[0].values
    )

chunk_embeddings = np.array(
    chunk_embeddings,
    dtype=np.float32
)
# "all-MiniLM-L6-v2" is a popular, lightweight embedding model.
# It produces 384-dimensional vectors and runs fast even on CPU.
# On first run, it downloads ~30MB. After that, it uses the cached version.
# In production, you might use larger models for better accuracy:
#   - "all-mpnet-base-v2" (420MB, more accurate)
#   - OpenAI's text-embedding-3-small (API-based, no local model needed)

print("Creating embeddings for document chunks...")

# chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
# .encode() converts each chunk text into a 384-dimensional vector.
# show_progress_bar=True shows a progress bar as it processes.
# The result is a NumPy array of shape (num_chunks, 384).
# Each row is one chunk's embedding vector.

# Convert to float32 — FAISS requires this specific data type.
chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

# Normalize the vectors for cosine similarity search.
# Normalization scales each vector to have length 1.
# After normalization, "inner product" (dot product) = cosine similarity.
faiss.normalize_L2(chunk_embeddings)

# Build the FAISS index.
dimension = chunk_embeddings.shape[1]  # 384 dimensions for all-MiniLM-L6-v2.
# .shape returns (num_chunks, 384). [1] gets the second value = 384.

index = faiss.IndexFlatIP(dimension)
# IndexFlatIP = "Flat Index with Inner Product" search.
# "Flat" means it does an exact search (checks every vector). Best for small datasets.
# "IP" means Inner Product — combined with normalized vectors, this gives cosine similarity.
# For billions of vectors, you'd use IndexIVFFlat or IndexHNSW for approximate but faster search.

index.add(chunk_embeddings)
# .add() stores all the chunk vectors in the FAISS index.

# Save FAISS index
faiss.write_index(index, "index.faiss")
print("FAISS index saved to index.faiss")

# Save chunks

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved to chunks.pkl")
# Now we can search for similar vectors extremely fast.



print(f"FAISS index built with {index.ntotal} vectors ({dimension} dimensions each).")
print("Ready to answer questions!\n")




