import os
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document  # Used to create Document objects from reviews
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # For embeddings
from langchain_astradb import AstraDBVectorStore  # Vector store to store/retrieve embeddings
from langchain_groq import ChatGroq  # LLM model interface
from langchain.chains import create_retrieval_chain  # Create retrieval-based chain
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combine multiple docs
from langchain_core.prompts import MessagesPlaceholder  # Placeholder for chat history in prompts
from langchain.chains import create_history_aware_retriever  # Wrap retriever with chat history context
from langchain.prompts import ChatPromptTemplate  # For constructing prompts
from langchain_community.chat_message_histories import ChatMessageHistory  # Stores chat history
from langchain_core.chat_history import BaseChatMessageHistory  # Base class for chat history
from langchain_core.runnables.history import RunnableWithMessageHistory  # Chain with memory support

# Load environment variables from .env file
load_dotenv()
astradb_token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
keyspace = os.environ.get("ASTRA_DB_KEYSPACE")
api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")

# Ensure all necessary environment variables are set
if not astradb_token or not keyspace or not api_endpoint:
    raise ValueError("Missing AstraDB environment variables. Check your .env file.")

# Initialize embeddings model
embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-base-en-v1.5")

# Load dataset containing product titles and reviews
df = pd.read_csv("headphones_dataset.csv")
data = df[["product_title", "review"]]

# Convert each review into a Document object with metadata
docs = []
for _, row in data.iterrows():
    metadata = {"product_name": row["product_title"]}
    doc = Document(page_content=row["review"], metadata=metadata)
    docs.append(doc)

# Initialize AstraDB vector store to store embeddings
vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="flipkart_collection",  # Name of the collection
    token=astradb_token,                     # Astra DB token
    namespace=keyspace                       # Keyspace/namespace
)

# vstore.add_documents(docs)

# Initialize LLM model (ChatGroq)
model = ChatGroq(model='llama-3.1-8b-instant', temperature=0.3)

# Prompt to reformulate queries using chat history
retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

# Convert the vector store into a retriever with top-3 results
retriever = vstore.as_retriever(search_kwargs={"k": 3})

# Create a prompt template to contextualize queries with chat history
contextualize_query_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Wrap retriever to make it history-aware
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_query_prompt
)

# QA Prompt Template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Use the given context (product reviews) to answer the user's question. "
     "If the context does not contain the answer, say 'I could not find this in the reviews.'"),
    ("human", "Question: {input}\n\nContext:\n{context}\n\nAnswer:")
])

# Create a chain to combine multiple document contexts for answering
qa_chain = create_stuff_documents_chain(model, qa_prompt)

# Combine retriever and QA chain into a single retrieval-based chain
chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Dictionary to store session chat histories
store = {}

# Function to get or create a chat history for a session
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the chain with memory support (stores chat history automatically)
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Example usage: ask a question and get an answer
if __name__ == "__main__":
    result = chain_with_memory.invoke(
        {"input": "Name the best Bluetooth buds with good bass?"},
        config={"configurable": {"session_id": "test_session"}}
    )
    print(result["answer"])