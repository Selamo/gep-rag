import os
from dotenv import load_dotenv

# Load .env and set USER_AGENT BEFORE importing LangChain
load_dotenv()
os.environ['USER_AGENT'] = os.getenv('USER_AGENT', 'gep-rag-system/1.0')

# NOW import LangChain modules (after USER_AGENT is set)
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_engine import get_vectorstore

def ingest_data():
    print("Starting ingestion...")
    
    documents = []
    
    # 1. Web Ingestion
    urls = ["https://gepprotech.com"]
    print(f"Loading from URLs: {urls}")
    try:
        web_loader = WebBaseLoader(urls)
        web_docs = web_loader.load()
        documents.extend(web_docs)
        print(f"✓ Loaded {len(web_docs)} documents from Web.")
    except Exception as e:
        print(f"✗ Error loading from Web: {e}")

    # 2. PDF Ingestion
    pdf_dir = "data"
    if os.path.exists(pdf_dir):
        print(f"Loading PDFs from: {pdf_dir}")
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_dir, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    documents.extend(pdf_docs)
                    print(f"✓ Loaded {len(pdf_docs)} pages from {filename}")
                except Exception as e:
                    print(f"✗ Error loading PDF {filename}: {e}")
    else:
        print(f"⚠ Directory '{pdf_dir}' not found. Skipping PDF ingestion.")

    if not documents:
        print("✗ No documents to ingest.")
        return

    # 3. Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(splits)} chunks.")

    # 4. Vector Store Upsert
    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(documents=splits)
        print("✓ Successfully added documents to Vector Store.")
    except Exception as e:
        print(f"✗ Error adding to Vector Store: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("GEP PROTECH RAG - DATA INGESTION")
    print("=" * 60)
    ingest_data()