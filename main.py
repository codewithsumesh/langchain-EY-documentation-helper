#Any contains any type of data types
from typing import Any

import streamlit as st

from backend.core import run_llm


#extract source urls from retrived documents k =4 so 4 urls
def _format_sources(context_docs: list[Any]) -> list[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]
#this controls the browser page
'''Streamlit is a Python framework for quickly building a web UI for your Python/AI application.'''
st.set_page_config(page_title="EY Documentation Helper", layout="centered")
st.title("EY Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        '''session state is where our application remembers the conversation history--storage'''
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "🤖 Ask me anything about EY. I'll retrieve relevant content and cite sources.",
            "sources": [],#creates a collapsible sources section
        }
    ]
#display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…") # user input
if prompt:#only after submit
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []}) #saves user messages
    with st.chat_message("user"):
        st.markdown(prompt)#display user messages
#calling rag background
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):#loading
                result: dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)" #extract answer
                sources = _format_sources(result.get("context", [])) #extract sources

            st.markdown(answer)#show answer
            if sources:
                with st.expander("Sources"):#show sources all
                    for s in sources:
                        st.markdown(f"- {s}")
#save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)