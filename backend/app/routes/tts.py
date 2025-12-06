from fastapi import APIRouter, HTTPException, Response
from app.model.schemas import TTSRequest
from app.services.tts_service import TTSService

router = APIRouter()

@router.get("/voices")
async def get_voices():
    """Get list of available voices"""
    return {"voices": TTSService.list_voices()}

@router.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate audio from text"""
    try:
        audio_content = TTSService.generate_audio(request.text, request.voice)
        return Response(content=audio_content, media_type="audio/mpeg")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
