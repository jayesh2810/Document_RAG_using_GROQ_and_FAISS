# RAG-Hybrid-Search

An AI-powered **Retrieval-Augmented Generation (RAG) pipeline** that enables **document search and interaction** using **FAISS and SentenceTransformers** for hybrid retrieval and **Groq's LLM** for generating contextual answers. It allows users to query and chat about indexed documents efficiently via a Streamlit interface.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jayesh2810/Document_RAG_using_GROQ_and_FAISS.git
   cd Document_RAG_using_GROQ_and_FAISS

   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt

   ```

3. **Create .env file in the project directory and add your Groq API key:**

   ```bash
   GROQ_API_KEY=your_api_key_here

   ```

4. **Run the script:**
   ```bash
   streamlit run Document_RAG.py
   ```
