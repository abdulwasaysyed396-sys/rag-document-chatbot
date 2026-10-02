import os
import numpy as np
import faiss
import gradio as gr

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "openai/gpt-oss-120b"

client = InferenceClient(token=HF_TOKEN)

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global document data
chunks = []
index = None


# -------------------------
# Extract text from PDF
# -------------------------

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# -------------------------
# Split text into chunks
# -------------------------

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -------------------------
# Process uploaded PDF
# -------------------------

def process_document(pdf_path):

    global chunks, index

    if pdf_path is None:
        return "Please upload a PDF first."

    text = extract_text(pdf_path)

    if not text.strip():
        return "Could not extract text from this PDF."

    chunks = chunk_text(text)

    # Create embeddings
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS vector database
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Store embeddings
    index.add(embeddings)

    return f"Document processed successfully! Created {len(chunks)} chunks."


# -------------------------
# Ask question
# -------------------------

def ask_question(question):

    if index is None:
        return "Please upload and process a PDF first."

    if not question.strip():
        return "Please enter a question."

    # Embed the question
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # Retrieve top 3 relevant chunks
    k = min(3, len(chunks))

    scores, results = index.search(
        question_embedding,
        k
    )

    # Build context
    context = "\n\n".join(
        chunks[i] for i in results[0]
    )

    # Send retrieved context to LLM
    prompt = f"""
Answer the question using ONLY the provided document context.

If the answer cannot be found in the context, say:
"I could not find the answer in the document."

Document context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer questions using only the provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=400,
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(title="Chat With Your Documents") as demo:

    gr.Markdown("# 📚 Chat With Your Documents")
    gr.Markdown(
        "Upload a PDF, process it, and ask questions about its contents."
    )

    pdf = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button("Process Document")

    status = gr.Textbox(
        label="Status"
    )

    process_button.click(
        process_document,
        inputs=pdf,
        outputs=status
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What is this document about?"
    )

    ask_button = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        lines=8
    )

    ask_button.click(
        ask_question,
        inputs=question,
        outputs=answer
    )


demo.launch()