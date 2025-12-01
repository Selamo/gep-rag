from fastapi import APIRouter, HTTPException
from app.model.schemas import IngestResponse
from app.services.ingestion_service import ingest_data

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint():
    try:
        ingest_data()
        return IngestResponse(status="success", message="Ingestion completed successfully")
    except Exception as e:
        return IngestResponse(status="error", message=str(e))
