from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS, cross_origin
import torch
from transformers import AutoModel, AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
ALLOWED_ORIGIN = os.getenv('ALLOWED_ORIGIN')
CORS(app, supports_credentials=True, origins=[ALLOWED_ORIGIN])

# Add CORS headers once in after_request
@app.after_request
def add_cors_headers(response):
    if 'Access-Control-Allow-Origin' in response.headers:
        del response.headers['Access-Control-Allow-Origin']
    if 'Access-Control-Allow-Headers' in response.headers:
        del response.headers['Access-Control-Allow-Headers']  
    if 'Access-Control-Allow-Methods' in response.headers:
        del response.headers['Access-Control-Allow-Methods']
    
    # Set new headers with specific origin instead of wildcard
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGIN
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,X-Requested-With,Accept,Origin'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

@app.route('/api/AI/check-toxic', methods=['OPTIONS'])
def options_handler():
    response = make_response()
    return response

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

@app.route("/api/AI/check-toxic", methods=["POST"])
@cross_origin()
def analyze_comment():
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if "content" not in data:
        return jsonify({"error": "No content provided in the request"}), 400
    
    comment = data["content"]
    
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
    threshold = 0.8
    is_toxic = any(score > threshold for score in results.values())
    
    return jsonify({
        "payload": {
            "is_toxic": is_toxic,
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)