import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableConfig
from recommender import chain_with_memmory, store, get_session_history  # Import from the new file

st.title("E-commerce Product Recommendation Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

def get_session_history_streamlit(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

for message in st.session_state.chat_history.messages:
    with st.chat_message(message.type):
        st.write(message.content)

if prompt := st.chat_input("Ask me about products..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    st.session_state.chat_history.add_user_message(prompt)
    
    config = RunnableConfig(configurable={"session_id": st.session_state.session_id})
    response = chain_with_memmory.invoke({"input": prompt}, config=config)
    
    with st.chat_message("assistant"):
        st.write(response["answer"])
    
    st.session_state.chat_history.add_ai_message(response["answer"])

if st.button("Clear Chat"):
    st.session_state.chat_history.clear()
    st.rerun()  # Updated from st.experimental_rerun()