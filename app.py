from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app) 

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

# Initialize global variables
model = None
tokenizer = None
label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Load model function
def load_model_and_tokenizer():
    global model, tokenizer
    
    # Load PhoBERT model and tokenizer
    phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    
    # Initialize the model
    model = PhoBertModel(phobert)
    
    # Load weights
    weights_path = "weight/phoBertModel_weights_50k_8_new.pth"
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

load_model_and_tokenizer()

@app.route("/")
def hello_world():
    return render_template("cors_test.html")

@app.route("/api/reports", methods=["POST"])
def analyze_comment():
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Handle TReport format with content, type, and uuid
    if "content" not in data:
        return jsonify({"error": "No content provided in the request"}), 400
    
    comment = data["content"]
    report_type = data.get("type", "unknown")
    report_uuid = data.get("uuid", "")
    
    # Tokenize the input
    inputs = tokenizer(comment, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Process results
    results = {}
    for i, label in enumerate(label_names):
        score = outputs[0][i].item()
        results[label] = score
        
    # Determine if comment is toxic (if any category exceeds threshold)
    threshold = 0.5
    is_toxic = any(score > threshold for score in results.values())
    
    return jsonify({
        "uuid": report_uuid,
        "content": comment,
        "type": report_type,
        "is_toxic": is_toxic,
        "scores": results
    })

if __name__ == "__main__":
    app.run(debug=True)