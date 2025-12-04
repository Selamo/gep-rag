import os
from app.config import settings

os.environ['USER_AGENT'] = settings.USER_AGENT

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.rag_service import get_vectorstore

def ingest_data():
    print("Starting ingestion...")
    
    documents = []
    
    # 1. Web Ingestion
    urls = ["https://gepprotech.com"]
            # ,"https://gepprotech.com/home","https://gepprotech.com/about"
            #"https://gepprotech.com/courses","https://gepprotech.com/tutors","https://gepprotech.com/gallery",
            #"https://gepprotech.com/achievements","https://gepprotech.com/contact","https://gepprotech.com/enroll"]
    print(f"Loading from URLs: {urls}")
    try:
        web_loader = WebBaseLoader(urls)
        web_docs = web_loader.load()
        if web_docs:
            documents.extend(web_docs)
            print(f"✓ Loaded {len(web_docs)} documents from Web.")
            # print(f"  Sample content: {web_docs[0].page_content[:100]}...")
        else:
            print("⚠ Web loader returned no documents.")
    except Exception as e:
        print(f"✗ Error loading from Web: {e}")

    # 2. PDF Ingestion
    possible_paths = [
        "data",
        "../data",
        "../../data",
        os.path.join(os.path.dirname(__file__), "../../../data")
    ]
    
    pdf_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            pdf_dir = path
            break
            
    if pdf_dir:
        print(f"Loading PDFs from: {os.path.abspath(pdf_dir)}")
        pdf_count = 0
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_dir, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    documents.extend(pdf_docs)
                    pdf_count += 1
                    print(f"✓ Loaded {len(pdf_docs)} pages from {filename}")
                except Exception as e:
                    print(f"✗ Error loading PDF {filename}: {e}")
        if pdf_count == 0:
            print("⚠ No PDF files found in data directory.")
    else:
        print(f"⚠ Directory 'data' not found. Checked: {possible_paths}. Skipping PDF ingestion.")

    if not documents:
        print("✗ No documents to ingest.")
        return

    print(f"Total documents to process: {len(documents)}")

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
