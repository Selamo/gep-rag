import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from app.config import settings

# Ensure USER_AGENT is set for LangChain
os.environ['USER_AGENT'] = settings.USER_AGENT

def get_llm():
    """Get Google Gemini LLM for chat responses"""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.7
    )

def get_embeddings():
    """Get Hugging Face embeddings (free, no quota limits)"""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_vectorstore():
    """Initialize or connect to Pinecone vector store"""
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    embeddings = get_embeddings()
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    index_name = settings.PINECONE_INDEX_NAME
    
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"✓ Index '{index_name}' created successfully!")
    else:
        print(f"✓ Using existing index '{index_name}'")
    
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)
