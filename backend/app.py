# backend/app.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import json
import traceback
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from dotenv import load_dotenv

# --- LangChain / Groq imports ---
from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
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

# For first deployment, allow all origins.
# After you know your Vercel URL, change origins to that exact domain.
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
groq_api_key = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_FOLDER = os.path.join(BASE_DIR, "temp_files")
os.makedirs(TEMP_FOLDER, exist_ok=True)
print(f"✅ Temporary file directory: {TEMP_FOLDER}")

# ============================================================
# 2. NLP Model (Aura Symptom)
# ============================================================

NLP_MODEL_PATH = os.path.join(BASE_DIR, "models", "nlp_model")
nlp_model = None
nlp_tokenizer = None
id_to_label = {}

try:
    nlp_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    nlp_model = DistilBertForSequenceClassification.from_pretrained(NLP_MODEL_PATH)
    nlp_model.to(device)
    nlp_model.eval()

    with open(os.path.join(NLP_MODEL_PATH, "label_mapping.json"), "r") as f:
        id_to_label_str_keys = json.load(f)
        id_to_label = {int(k): v for k, v in id_to_label_str_keys.items()}

    print("✅ NLP model, tokenizer, and label mapping loaded.")
except Exception as e:
    print(f"❌ Error loading NLP model: {e}")

# ============================================================
# 3. EEG Model
# ============================================================

class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[9, 19, 39],
        bottleneck_channels=32,
        use_residual=True,
        dropout=0.2,
    ):
        super().__init__()
        self.use_residual = use_residual

        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else None
        )

        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels if self.bottleneck else in_channels,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        self.batchnorm = nn.BatchNorm1d(out_channels * len(kernel_sizes))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if use_residual:
            self.residual = nn.Conv1d(
                in_channels, out_channels * len(kernel_sizes), kernel_size=1
            )

    def forward(self, x):
        x_in = self.bottleneck(x) if self.bottleneck else x
        out = torch.cat([conv(x_in) for conv in self.conv_list], dim=1)
        out = self.batchnorm(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.use_residual:
            out = out + self.residual(x)

        return out


class InceptionTime(nn.Module):
    def __init__(self, in_channels, num_classes, num_blocks=2, out_channels=16):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else out_channels * 3
            blocks.append(InceptionBlock1D(in_ch, out_channels))

        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 3, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


EEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "eeg_model", "inception_full_model.pth")
eeg_model = None

try:
    eeg_model = torch.load(EEG_MODEL_PATH, map_location=device, weights_only=False)
    eeg_model.to(device)
    eeg_model.eval()
    print("✅ EEG model loaded.")
except Exception as e:
    print(f"❌ Error loading EEG model: {e}")

# ============================================================
# 4. RAG Chatbot Setup
# ============================================================

vector_store = None
try:
    KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base", "epilepsy_knowledgebase.txt")
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=100
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embeddings)

    print("✅ Chatbot vector store created.")
except Exception as e:
    print(f"❌ Error creating vector store: {e}")

# ============================================================
# 5. Chatbot Streaming
# ============================================================

def stream_chatbot_response(query: str):
    try:
        callbacks = [StreamingStdOutCallbackHandler()]

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            streaming=True,
            callbacks=callbacks,
        )

        retriever = vector_store.as_retriever()

        prompt_template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know. Keep the answer concise.
Context: {context}
Question: {question}
Helpful Answer:"""

        qa_prompt = PromptTemplate.from_template(prompt_template)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt},
        )

        for chunk in qa_chain.stream({"query": query}):
            if "result" in chunk:
                yield chunk["result"]

    except Exception as e:
        traceback.print_exc()
        yield f"An error occurred: {str(e)}"


# ============================================================
# 6. EEG Helper Functions
# ============================================================

def create_windows(data, fs=256, window_sec=2, overlap=0.5):
    n_samples = data.shape[1]
    win_size = int(window_sec * fs)
    step = int(win_size * (1 - overlap))

    windows = []
    for start in range(0, n_samples - win_size + 1, step):
        win = data[:, start : start + win_size]
        windows.append(win)

    return np.array(windows)


def preprocess_eeg_file(file_path):
    CHANNELS_TO_USE = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ", "P7-T7", "O1-O2",
        "T7-FT9", "FT10-T8",
    ]
    TARGET_FS = 256

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.strip())

    existing_channels = [ch for ch in CHANNELS_TO_USE if ch in raw.ch_names]
    if not existing_channels:
        raise ValueError("No valid EEG channels found in the file.")

    raw.pick(existing_channels)

    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto")

    data = raw.get_data()
    windows = create_windows(data)

    return windows.astype(np.float32)


def predict_from_eeg(file_storage):
    if eeg_model is None:
        return "EEG model is not available."

    temp_file_path = os.path.join(TEMP_FOLDER, file_storage.filename)

    try:
        file_storage.save(temp_file_path)
        windows = preprocess_eeg_file(temp_file_path)

        if windows.shape[0] == 0:
            return "Could not extract valid data windows from the EEG file."

        windows_tensor = torch.from_numpy(windows).to(device)

        with torch.no_grad():
            outputs = eeg_model(windows_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        seizure_predictions = np.sum(preds == 1)
        total_windows = len(preds)
        seizure_ratio = seizure_predictions / total_windows

        if seizure_ratio > 0.5:
            return f"Seizure Risk Detected ({seizure_predictions}/{total_windows} windows positive)"
        else:
            return f"No Seizure Risk Detected ({seizure_predictions}/{total_windows} windows positive)"

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ============================================================
# 7. API Endpoints
# ============================================================

@app.route("/predict_aura", methods=["POST"])
def predict_aura_endpoint():
    if nlp_model is None:
        return jsonify({"error": "NLP model is not available."}), 500

    data = request.get_json()
    if not data or "symptom_text" not in data:
        return jsonify({"error": 'Invalid input, "symptom_text" is required.'}), 400

    symptom = data["symptom_text"]

    encoding = nlp_tokenizer(
        symptom,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    ).to(device)

    with torch.no_grad():
        outputs = nlp_model(**encoding)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=1).item()
    probabilities = F.softmax(logits, dim=1)[0]

    combined_label = id_to_label.get(predicted_id, "Unknown_Error")
    aura_stage, seizure_risk_text = combined_label.rsplit("_", 1)

    yes_risk_id = next(
        (key for key, value in id_to_label.items() if value == f"{aura_stage}_Yes"),
        -1,
    )
    seizure_risk_prob = probabilities[yes_risk_id].item() if yes_risk_id != -1 else 0.0

    if seizure_risk_prob > 0.80:
        risk_message = "High Risk"
    elif seizure_risk_prob > 0.60:
        risk_message = "Medium Risk"
    else:
        risk_message = "Low Risk"

    return jsonify(
        {
            "predicted_aura_stage": aura_stage.replace("_", " "),
            "predicted_seizure_risk": risk_message,
        }
    )


@app.route("/predict_eeg", methods=["POST"])
def predict_eeg_endpoint():
    if "eeg_file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["eeg_file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.endswith(".edf"):
        return jsonify({"error": "Invalid file type. Please upload a .edf file."}), 400

    try:
        prediction_result = predict_from_eeg(file)
        return jsonify({"prediction": prediction_result})
    except Exception as e:
        traceback.print_exc()
        return (
            jsonify({"error": f"An error occurred during processing: {str(e)}"}),
            500,
        )


@app.route("/ask_chatbot", methods=["POST"])
def ask_chatbot_endpoint():
    if not groq_api_key:
        return Response("Error: GROQ_API_KEY is not configured.", status=500)

    if vector_store is None:
        return Response("Error: Knowledge base not available.", status=500)

    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return Response("Please ask a question.", status=400)

    return Response(stream_chatbot_response(query), mimetype="text/plain")


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


# ============================================================
# 8. Main (local dev only)
# ============================================================

if __name__ == "__main__":
    # For Render, gunicorn will import `app` and use PORT env var.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
