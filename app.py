from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModel, AutoTokenizer
import os
import threading
import logging
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# In production, CORS is handled entirely by nginx (single source of truth).
# In development (no nginx), set ALLOWED_ORIGIN in .env to enable Flask-CORS.
_allowed_origin = os.getenv("ALLOWED_ORIGIN")
if _allowed_origin:
    from flask_cors import CORS
    CORS(
        app,
        resources={r"/api/*": {"origins": _allowed_origin}},
        supports_credentials=True,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )


# Define PhoBERT model class
class PhoBertModel(torch.nn.Module):
    def __init__(self, phobert):
        super(PhoBertModel, self).__init__()
        self.bert = phobert
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 6)  # 6 classes

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden_state, output_1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooler = self.pre_classifier(output_1)
        activation_1 = torch.nn.Tanh()(pooler)

        drop = self.dropout(activation_1)
        output_2 = self.classifier(drop)
        output = torch.nn.Sigmoid()(output_2)
        return output

logger = logging.getLogger(__name__)

model = None
tokenizer = None
_model_ready = threading.Event()
label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_model_and_tokenizer():
    global model, tokenizer
    try:
        logger.info("Loading PhoBERT model...")
        phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

        model = PhoBertModel(phobert)

        weights_path = "weight/phoBertModel_weights_50k_8_new.pth"
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False)
        )
        model.eval()
        _model_ready.set()
        logger.info("Model loaded and ready.")
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)


# Load in background so gunicorn workers accept requests immediately.
threading.Thread(target=load_model_and_tokenizer, daemon=True).start()

@app.route("/")
def hello_world():
    return render_template("cors_test.html")


@app.route("/api/AI/health")
def health():
    if _model_ready.is_set():
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "loading"}), 503


@app.route("/api/AI/check-toxic", methods=["POST"])
def analyze_comment():
    if not _model_ready.is_set():
        return jsonify({"error": "Model is still loading, please try again shortly"}), 503

    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    if "content" not in data:
        return jsonify({"error": "No content provided in the request"}), 400

    comment = data["content"]

    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])

    results = {label: outputs[0][i].item() for i, label in enumerate(label_names)}

    threshold = 0.8
    is_toxic = any(score > threshold for score in results.values())

    return jsonify({
        "payload": {
            "is_toxic": is_toxic,
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv('PORT'))