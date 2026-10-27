from fastapi import APIRouter, UploadFile, File
from typing import List
from modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger


router=APIRouter()

@router.post("/upload_pdfs/")
def upload_pdfs(files:List[UploadFile] = File(...)):
    try:
        logger.info("Recieved uploaded files")
        chunk_count=load_vectorstore(files)
        logger.info("Document added to vectorstore")
        return {"messages":"Files processed and vectorstore updated","chunks":chunk_count}
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500,content={"error":str(e)})
