# streamlit_app.py
import streamlit as st
import requests

st.title("RAG + Ollama Chatbot")

query = st.text_input("질문을 입력하세요:")

if st.button("검색"):
    resp = requests.post("http://127.0.0.1:8000/rag", params={"query": query})
    data = resp.json()
    st.subheader("🔍 검색된 문서")
    for c in data["context"]:
        st.write("- ", c)

    st.subheader("🤖 답변")
    st.write(data["answer"])
