from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# llama3-70b-8192 was decommissioned by Groq; override with GROQ_MODEL if needed
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.

Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.

---

🔍 **Context**:
{context}

🙋‍♂️ **User Question**:
{question}

---

💬 **Answer**:
- Respond in a calm, factual, and respectful tone.
- Use simple explanations when needed.
- If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT make up facts.
- Do NOT give medical advice or diagnoses.
"""
)


def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=GROQ_MODEL,
        temperature=0
    )


def get_llm_chain(llm=None):
    return prompt | (llm or get_llm()) | StrOutputParser()
