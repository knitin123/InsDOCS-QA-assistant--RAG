from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.load_vectorstore import retrieve
from modules.query_handlers import query_chain
from logger import logger

router=APIRouter()

# plain def: FastAPI runs it in a worker thread, so the blocking
# embedding/Pinecone/Groq calls don't stall the event loop
@router.post("/ask/")
def ask_question(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        docs = retrieve(question, top_k=3)
        chain = get_llm_chain()
        result = query_chain(chain, question, docs)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
