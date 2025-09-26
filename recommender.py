import os
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever

# Load environment variables
load_dotenv()

# Initialize global store for session history
store = {}

# Fetch environment variables
astradb_token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
keyspace = os.environ.get("ASTRA_DB_KEYSPACE")
api_endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")

# Initialize embeddings
embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-base-en-v1.5")

# Initialize vector store
vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="flipkart_collection",
    token=astradb_token,
    namespace=keyspace,
    api_endpoint=api_endpoint
)

# Load and prepare data (example function to convert CSV to documents)
def prepare_documents():
    df = pd.read_csv("headphones_dataset.csv")
    data = df[["product_title", "review"]]
    product_list = []
    for index, row in data.iterrows():
        object = {
            "product_name": row["product_title"],
            "review": row["review"]
        }
        product_list.append(object)
    
    docs = []
    for object in product_list:
        metadata = {"product_name": object["product_name"]}
        page_content = object["review"]
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    
    # Add documents to vector store
    vstore.add_documents(docs)
    return vstore

# Prepare documents on module load
vstore = prepare_documents()

# Initialize model
model = ChatGroq(model='llama-3.1-8b-instant', temperature=0.3)

# Define retriever prompt
retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

# Define contextualize query prompt
contextualize_query_prompt = ChatPromptTemplate.from_messages([
    ('system', retriever_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

# Create history-aware retriever
retriever = vstore.as_retriever(search_kwargs={"k": 3})
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_query_prompt)

# Define QA bot template
QA_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:
    """

# Define QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ('system', QA_BOT_TEMPLATE),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')
])

# Create QA chain
qa_chain = create_stuff_documents_chain(model, qa_prompt)

# Create retrieval chain
chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Define session history function
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create chain with memory
chain_with_memmory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

if __name__ == "__main__":
    # Example test (optional)
    result = chain_with_memmory.invoke(
        {"input": "What are the best Bluetooth buds?"},
        config={"configurable": {"session_id": "test_session"}}
    )
    print(result["answer"])