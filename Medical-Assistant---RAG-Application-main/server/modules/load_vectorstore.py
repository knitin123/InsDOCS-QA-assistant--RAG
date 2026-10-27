import os
import time
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from modules.pdf_handlers import save_uploaded_files
from logger import logger

load_dotenv()

PINECONE_ENV=os.getenv("PINECONE_ENV","us-east-1")
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME","insightdocs")
# models/embedding-001 has been retired by Google; gemini-embedding-001 truncated to 768 dims
EMBED_MODEL=os.getenv("EMBED_MODEL","models/gemini-embedding-001")
EMBED_DIM=768

CHUNK_SIZE=500
CHUNK_OVERLAP=50
UPSERT_BATCH_SIZE=100


# clients are created lazily (not at import time) so the app and tests
# can start without network access or API keys

@lru_cache
def get_embed_model():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL,output_dimensionality=EMBED_DIM)


@lru_cache
def get_index():
    pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    return pc.Index(PINECONE_INDEX_NAME)


def clean_metadata(metadata:dict)->dict:
    # Pinecone metadata only accepts strings, numbers, booleans and lists of strings
    return {k:v for k,v in metadata.items() if isinstance(v,(str,int,float,bool,list))}


def chunk_pdf(file_path:str)->list[Document]:
    documents=[
        Document(page_content=page.extract_text() or "",metadata={"source":file_path,"page":page_number})
        for page_number,page in enumerate(PdfReader(file_path).pages)
    ]
    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)


# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files)->int:
    embed_model=get_embed_model()
    index=get_index()
    total=0

    for file_path in save_uploaded_files(uploaded_files):
        chunks=chunk_pdf(file_path)

        texts=[chunk.page_content for chunk in chunks]
        # store the chunk text alongside the vector so it can be returned at query time
        metadatas=[{**clean_metadata(chunk.metadata),"text":chunk.page_content} for chunk in chunks]
        ids=[f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]

        logger.info(f"Embedding {len(texts)} chunks from {file_path}")
        embeddings=embed_model.embed_documents(texts)

        vectors=list(zip(ids,embeddings,metadatas))
        for start in range(0,len(vectors),UPSERT_BATCH_SIZE):
            index.upsert(vectors=vectors[start:start+UPSERT_BATCH_SIZE])

        logger.info(f"Upload complete for {file_path}")
        total+=len(chunks)

    return total


def retrieve(question:str,top_k:int=3)->list[Document]:
    embedded_query=get_embed_model().embed_query(question)
    res=get_index().query(vector=embedded_query,top_k=top_k,include_metadata=True)
    return [
        Document(page_content=match["metadata"].get("text",""),metadata=match["metadata"])
        for match in res["matches"]
    ]
