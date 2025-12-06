from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    success: bool
    error: str = None

class IngestResponse(BaseModel):
    status: str
    message: str

class TTSRequest(BaseModel):
    text: str
    voice: str
