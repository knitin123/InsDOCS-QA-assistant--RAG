from pathlib import Path
from langchain_core.documents import Document
from logger import logger

NO_CONTEXT_ANSWER="I'm sorry, but I couldn't find relevant information in the provided documents."

def query_chain(chain,user_input:str,docs:list[Document]):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        if not docs:
            return {"response":NO_CONTEXT_ANSWER,"sources":[]}

        context="\n\n".join(doc.page_content for doc in docs)
        answer=chain.invoke({"context":context,"question":user_input})

        # unique source file names, in retrieval order
        sources=list(dict.fromkeys(Path(doc.metadata.get("source","")).name for doc in docs))
        response={
            "response":answer,
            "sources":[s for s in sources if s]
        }
        logger.debug(f"Chain response:{response}")
        return response
    except Exception as e:
        logger.exception("Error on query chain")
        raise
