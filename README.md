# ShopSmart RAGBot

## Overview
ShopSmart RAGBot is an e-commerce chatbot built using Retrieval-Augmented Generation (RAG) with LangChain and Astra DB. It leverages Flipkart customer reviews to provide personalized product recommendations. Integrated with HuggingFace embeddings and Groq’s llama-3.1-8b-instant LLM, it’s deployed via a Streamlit web app with session management for a seamless user experience.

## Features
- Personalized product recommendations based on customer reviews.
- Real-time, context-aware responses using RAG technology.
- Interactive Streamlit interface with chat history and session support.
- Optimized vector store integration with Astra DB for accurate retrieval.

## Prerequisites
- Python 3.10+
- Required libraries: `streamlit`, `langchain`, `langchain-astradb`, `langchain-groq`, `langchain-huggingface`, `pandas`, `python-dotenv`
- Astra DB account and application token
- Groq API key
- Flipkart product review CSV file

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Md-Tauhid101/ShopSmart-RAGBot.git
   cd shopsmart-ragbot
   ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a .env file with your credentials:
    ```bash
    ASTRA_DB_APPLICATION_TOKEN=your_token
    ASTRA_DB_KEYSPACE=your_keyspace
    ASTRA_DB_API_ENDPOINT=your_endpoint
    GROQ_API_KEY=your_groq_key
    ```

4. Place flipkart_product_review.csv in the project directory or adjust the path in recommender.py.

5. Run the data loader:
    ```bash
    python recommender.py
    ```

6. Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```


### Local Setup
- On your local machine, replace the placeholders in the `.env` file with your actual values before running the project:

