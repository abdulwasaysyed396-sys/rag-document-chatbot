# Chat With Your Documents

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its contents.

## RAG Pipeline

PDF → Text Extraction → Chunking → Embeddings → FAISS Vector Database → Retrieval → LLM → Answer

## Technologies Used

- Python
- Gradio
- PyPDF
- Sentence Transformers
- FAISS
- Hugging Face
- python-dotenv

## Features

- Upload PDF documents
- Extract document text
- Split documents into chunks
- Generate embeddings
- Store embeddings in FAISS
- Retrieve relevant document sections
- Generate answers using an LLM
- Ground answers in the uploaded document

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
Create a .env file:

HF_TOKEN=your_token_here

Run:

python app.py

5. Scroll down.
6. Click **Commit changes**.

