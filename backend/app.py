# backend/app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import traceback

from dotenv import load_dotenv

# ---- LangChain / Groq ----
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# ============================================================
# 1. App + Environment
# ============================================================

load_dotenv()

app = Flask(__name__)
CORS(app)

groq_api_key = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 2. MODELS DISABLED (CLOUD SAFE)
# ============================================================

nlp_model = None
eeg_model = None

print("⚠️ NLP & EEG models disabled for cloud deployment.")
print("ℹ️ Full inference available in local environment only.")

# ============================================================
# 3. RAG CHATBOT SETUP (WORKS ON CLOUD)
# ============================================================

vector_store = None

try:
    kb_path = os.path.join(BASE_DIR, "knowledge_base", "epilepsy_knowledgebase.txt")

    with open(kb_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
    )

    docs = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(docs, embeddings)

    print("✅ Vector store created successfully")

except Exception as e:
    print("❌ Vector store error:", e)

# ============================================================
# 4. CHATBOT STREAMING
# ============================================================

def stream_chatbot_response(query: str):
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        retriever = vector_store.as_retriever()

        prompt = PromptTemplate.from_template(
            """Use the context to answer clearly.
If unsure, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""
        )

        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
        )

        for chunk in chain.stream({"query": query}):
            if "result" in chunk:
                yield chunk["result"]

    except Exception as e:
        traceback.print_exc()
        yield f"Error: {str(e)}"

# ============================================================
# 5. API ENDPOINTS
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/ask_chatbot", methods=["POST"])
def ask_chatbot():
    if not groq_api_key:
        return Response("GROQ_API_KEY missing", status=500)

    if vector_store is None:
        return Response("Knowledge base not loaded", status=500)

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return Response("Empty query", status=400)

    return Response(stream_chatbot_response(query), mimetype="text/plain")


@app.route("/predict_aura", methods=["POST"])
def predict_aura():
    return jsonify({
        "status": "disabled",
        "message": "Aura prediction disabled in cloud deployment. Run locally for inference."
    }), 200


@app.route("/predict_eeg", methods=["POST"])
def predict_eeg():
    return jsonify({
        "status": "disabled",
        "message": "EEG prediction disabled in cloud deployment. Run locally for inference."
    }), 200


# ============================================================
# 6. MAIN (LOCAL ONLY)
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
