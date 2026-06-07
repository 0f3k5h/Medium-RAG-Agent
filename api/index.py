from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

app = FastAPI()

# Initialize Clients
client = OpenAI(
    api_key=os.environ.get("LLMOD_API_KEY"),
    base_url="https://api.llmod.ai/v1" 
)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("medium-rag-index")

# The strict system prompt required by your assignment instructions
SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only based on the "
    "Medium articles dataset context provided to you (metadata and article passages). You must "
    "not use any external knowledge, the open internet, or information that is not explicitly "
    "contained in the retrieved context. If the answer cannot be determined from the provided "
    "context, respond: 'I don't know based on the provided Medium articles data.' Always explain "
    "your answer using the given context, quoting or paraphrasing the relevant article passage "
    "or metadata when helpful."
)

# Defines the expected JSON input format
class Query(BaseModel):
    question: str

@app.get("/api/stats")
def get_stats():
    # Returns the exact configuration you used during ingestion
    return {
        "chunk_size": 500,
        "overlap_ratio": 0.2,
        "top_k": 5
    }

@app.post("/api/prompt")
def generate_response(query: Query):
    # 1. Embed the user's incoming question
    question_embedding = client.embeddings.create(
        input=query.question,
        model="ZYRANGG-text-embedding-3-small"
    ).data[0].embedding

    # 2. Search Pinecone for the top 5 most relevant chunks
    search_results = index.query(
        vector=question_embedding,
        top_k=5, 
        include_metadata=True
    )

    # 3. Format the retrieved text to pass to the LLM and the JSON output
    context_text = ""
    context_output = []
    
    for match in search_results.matches:
        meta = match.metadata

        # Stitch together the text for the prompt
        context_text += (
                f"\nArticle ID: {meta.get('article_id')}"
                f"\nTitle: {meta.get('title')}"
                f"\nAuthor(s): {meta.get('authors')}"
                f"\nURL: {meta.get('url')}"
                f"\nTags: {meta.get('tags')}"
                f"\nPassage: {meta.get('chunk')}\n---"
            )
                    
        # Build the exact JSON array required by the assignment
        context_output.append({
            "article_id": meta['article_id'],
            "title": meta['title'],
            "chunk": meta['chunk'],
            "score": float(match.score)
        })

    # 4. Construct the final prompt
    user_prompt = f"Question: {query.question}\n\nContext:\n{context_text}"

    # 5. Call the LLMod.ai LLM with the strict guardrails
    llm_response = client.chat.completions.create(
        model="ZYRANGG-gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    final_answer = llm_response.choices[0].message.content

    # 6. Return the perfectly formatted JSON dictionary
    return {
        "response": final_answer,
        "context": context_output,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt
        }
    }