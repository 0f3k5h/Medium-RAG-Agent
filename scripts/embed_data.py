import pandas as pd
from openai import OpenAI
from pinecone import Pinecone
import os

# Initialize API clients
client = OpenAI(
    api_key=os.environ.get("LLMOD_API_KEY"),
    base_url="https://api.llmod.ai/v1" 
)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("medium-rag-index")

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="ZYRANGG-text-embedding-3-small"
    )
    return response.data[0].embedding

def process_data():
    print("Loading dataset...")
    
    # Read the csv
    df = pd.read_csv('data/medium-english-50mb.csv') 
    
    # Process chunks in batches
    batch_size = 100
    vectors_to_upsert = []
    total_chunks = 0

    print(f"Processing {len(df)} articles...")

    for idx, row in df.iterrows():

        # Simple character chunking (Approx 500 words per chunk)
        words = str(row['text']).split()
        
        # Final hyperparameters selected based on optimization research
        chunk_size = 500
        overlap = 100 
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if not chunk_text.strip(): continue
                
            # Generate the semantic vector
            vector = get_embedding(chunk_text)
            
            # Prepare payload with metadata
            chunk_id = f"article_{idx}_chunk_{i}"
            metadata = {
                "article_id": str(idx),
                "title": str(row.get('title', '')),
                "authors": str(row.get('authors', '')),
                "url": str(row.get('url', '')),
                "tags": str(row.get('tags', '')),
                "chunk": chunk_text
            }
            vectors_to_upsert.append((chunk_id, vector, metadata))
            total_chunks += 1
            
            # Batch Upsert to Pinecone
            if len(vectors_to_upsert) >= batch_size:
                print(f"Upserting batch of {batch_size} chunks...")
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
                
    # Upsert any remaining chunks
    if vectors_to_upsert:
        print(f"Upserting final batch of {len(vectors_to_upsert)} chunks...")
        index.upsert(vectors=vectors_to_upsert)
        
    print(f"Ingestion complete! Total chunks embedded and uploaded: {total_chunks}")

if __name__ == "__main__":
    process_data()