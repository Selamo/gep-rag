from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gep RAG System")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Gep RAG System API is running"}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        from rag_engine import get_llm, get_vectorstore
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        llm = get_llm()
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever()

        system_prompt = (
            "You are a Gep assistant for answering questions about Gep (gepprotech.com). "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": request.query})
        return {"response": response["answer"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ingest")
async def ingest_endpoint():
    try:
        from ingestion import ingest_data
        ingest_data()
        return {"status": "Ingestion completed successfully"}
    except Exception as e:
        return {"status": "Ingestion failed", "error": str(e)}
