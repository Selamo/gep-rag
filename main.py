from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Set USER_AGENT to avoid warnings (BEFORE importing LangChain modules)
os.environ['USER_AGENT'] = os.getenv('USER_AGENT', 'gep-rag-system/1.0')

# NOW import LangChain modules (after USER_AGENT is set)
from rag_engine import get_llm, get_vectorstore
from ingestion import ingest_data
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(
    title="GEP RAG System",
    description="RAG-based chatbot for GEP Protech",
    version="1.0.0"
)

# Add CORS middleware to allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with ["https://gepprotech.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
@app.get("/health")
async def root():
    return {
        "message": "GEP RAG System API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        llm = get_llm()
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        system_prompt = (
            "You are a GEP assistant for answering questions about GEP Protech (gepprotech.com). "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise and helpful."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": request.query})
        return {"answer": response["answer"], "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/ingest")
async def ingest_endpoint():
    try:
        ingest_data()
        return {"status": "success", "message": "Ingestion completed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# This is crucial for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)