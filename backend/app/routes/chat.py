from fastapi import APIRouter, HTTPException
from app.model.schemas import QueryRequest, ChatResponse
from app.chain.rag_chain import get_rag_chain

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": request.query})
        return ChatResponse(answer=response["answer"], success=True)
    except Exception as e:
        return ChatResponse(answer="", success=False, error=str(e))
