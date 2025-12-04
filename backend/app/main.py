from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from app.routes import chat, ingest

app = FastAPI(
    title="GEP RAG System",
    description="RAG-based chatbot for GEP Protech",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat.router)
app.include_router(ingest.router)

@app.get("/")
@app.get("/health")
async def root():
    return {
        "message": "GEP RAG System API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
