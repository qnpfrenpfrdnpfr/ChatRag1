from fastapi import FastAPI
from app.rag import search
from app.llm import get_llm

app = FastAPI()

llm = get_llm()

@app.post("/rag")
def rag_answer(query: str):
    docs = search(query, k=3)
    context = "\n\n".join(docs)
    
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question: {query}

    Answer:
    """

    answer = llm.invoke(prompt)
    return {"query": query, "context": docs, "answer": answer} # query는 질문, answer은 qwen3:4b모델이 생성한 대답 context는 rag~?