import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    USER_AGENT = os.getenv('USER_AGENT', 'gep-rag-system/1.0')
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "gep-rag")
    PORT = int(os.getenv("PORT", 8000))

settings = Config()
