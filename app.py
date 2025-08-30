from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load a small summarization model (fast & free-friendly)
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Input schema
class TextIn(BaseModel):
    text: str

# Health check
@app.get("/")
def read_root():
    return {"message": "Summarizer API is running!"}

# Summarization endpoint
@app.post("/summarize")
def summarize(data: TextIn):
    summary = summarizer(
        data.text, 
        max_length=80, 
        min_length=20, 
        do_sample=False
    )
    return {"summary": summary[0]['summary_text']}
