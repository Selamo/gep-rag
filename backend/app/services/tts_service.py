import requests
from app.config import settings

class TTSService:
    BASE_URL = "https://yarngpt.ai/api/v1/tts"
    AVAILABLE_VOICES = [
        "Idera", "Emma", "Zainab", "Osagie", "Wura", "Jude", 
        "Chinenye", "Tayo", "Regina", "Femi", "Adaora", "Umar", 
        "Mary", "Nonso", "Remi", "Adam"
    ]

    @staticmethod
    def list_voices():
        return TTSService.AVAILABLE_VOICES

    @staticmethod
    def generate_audio(text: str, voice: str) -> bytes:
        if voice not in TTSService.AVAILABLE_VOICES:
            raise ValueError(f"Voice '{voice}' is not supported.")
        
        headers = {
            "Authorization": f"Bearer {settings.YARNGPT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "voice": voice,
            "response_format": "mp3" 
        }
        
        response = requests.post(TTSService.BASE_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to generate audio: {response.text}")
