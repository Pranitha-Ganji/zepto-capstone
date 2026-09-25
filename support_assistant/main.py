"""
Zepto Support Assistant: FastAPI Service
Author: Zepto AI/ML Engineering Guild
Module: /support_assistant

Description:
Provides the REST API interface for Zepto's Support Assistant.
Exposes POST /ask accepting a Pydantic request model ({"query": str})
and returning a validated Pydantic response model (answer, sources, confidence).
"""

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from support_assistant.agent import query_assistant, AskResponse

app = FastAPI(
    title="Zepto Support Assistant API",
    description="Grounded GenAI Support Assistant serving Zepto policy information.",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str = Field(..., description="The user question to be answered by the assistant.", example="What is the delivery fee for orders below INR 149?")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Zepto Support Assistant",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "support_assistant"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Processes customer query through the LangGraph RAG workflow.
    Routes between policy retrieval and direct answers.
    Returns Pydantic validated AskResponse.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        response = query_assistant(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal graph execution error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("support_assistant.main:app", host="0.0.0.0", port=7860, reload=False)
