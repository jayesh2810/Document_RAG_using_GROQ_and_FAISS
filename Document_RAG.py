import streamlit as st
from groq import Groq
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load environment variables (if needed)
from dotenv import load_dotenv
load_dotenv()

# Initialize Groq API client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the FAISS index and documents
@st.cache_resource
def load_faiss_index_and_documents():
    index = faiss.read_index("faiss_index.index")
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

index, documents = load_faiss_index_and_documents()

# Load the BAAI small model for encoding queries
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('BAAI/bge-small-en')

model = load_embedding_model()

# Hybrid search function
def hybrid_search(query, part_name=None):
    if part_name:
        # If part name is provided, find the corresponding document
        for doc in documents:
            if part_name.lower() in doc["filename"].lower():
                return doc
        return None  # Return None if no matching document is found
    else:
        # If no part name is provided, perform semantic search
        query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = index.search(query_embedding, k=1)  # Retrieve the top 1 document
        return documents[indices[0][0]]  # Return the most relevant document

# Function to generate answers using Groq's model
def generate_answer_with_groq(query, document):
    # Prepare the messages for Groq's model
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided document."},
        {"role": "user", "content": f"Document: {document['content']}\n\nQuestion: {query}\nAnswer:"}
    ]

    # Send the messages to Groq's model
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Use the appropriate Groq model
        messages=messages,
        max_tokens=150,  # Adjust based on your needs
        temperature=1.2,  # Adjust for creativity vs. accuracy
    )

    # Extract and return the answer
    return response.choices[0].message.content.strip()

# Streamlit app
def main():
    st.title("RAG Pipeline with Hybrid Search and Groq")
    st.write("Ask a question, and the system will retrieve the most relevant document and generate an answer.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input for part name (optional)
    part_name = st.text_input("Enter part name (optional):")

    # Input for user query
    query = st.text_input("Enter your query:")

    # Button to submit the query
    if st.button("Submit"):
        if query:
            # Perform hybrid search
            document = hybrid_search(query, part_name)
            if document:
                # Generate answer using Groq's model
                answer = generate_answer_with_groq(query, document)
                # Add query and answer to chat history
                st.session_state.chat_history.append({"query": query, "answer": answer, "part_name": part_name})
            else:
                st.error("No relevant document found.")
        else:
            st.warning("Please enter a query.")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**Part Name:** {chat['part_name']}")
            st.write(f"**Query:** {chat['query']}")
            st.write(f"**Answer:** {chat['answer']}")
            st.write("---")

# Run the Streamlit app
if __name__ == "__main__":
    main()