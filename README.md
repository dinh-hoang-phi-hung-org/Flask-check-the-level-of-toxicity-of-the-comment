# Flask Toxicity Checker

This service is a Vietnamese comment toxicity detection API powered by a fine-tuned PhoBERT model.

The system receives a comment as input and returns a prediction indicating whether the content is toxic, using a multi-label classification approach across 6 toxicity categories.

---

## Tech Stack

| Technology                                               | Version  | Purpose                                 |
| -------------------------------------------------------- | -------- | --------------------------------------- |
| [Flask](https://flask.palletsprojects.com)               | 2.2.3    | Core web framework (Python)             |
| [PyTorch](https://pytorch.org)                           | >=1.10.0 | Deep learning inference engine          |
| [Transformers](https://huggingface.co/docs/transformers) | >=4.18.0 | Pre-trained model loading (HuggingFace) |
| [PhoBERT](https://huggingface.co/vinai/phobert-base-v2)  | base-v2  | Vietnamese language model (VinAI)       |
| [sentencepiece](https://github.com/google/sentencepiece) | >=0.1.96 | Tokenization support for PhoBERT        |

---

## API Endpoints

| Method | Endpoint              | Description                                             |
| ------ | --------------------- | ------------------------------------------------------- |
| `POST` | `/api/AI/check-toxic` | Analyze a comment and return whether it is toxic or not |

### Request Body

```json
{
  "content": "nội dung bình luận cần kiểm tra"
}
```

### Response

```json
{
  "payload": {
    "is_toxic": true
  }
}
```

### Toxicity Labels

The model classifies comments across 6 categories. A comment is marked as toxic if any label exceeds the threshold of `0.8`:

| Label           | Description                  |
| --------------- | ---------------------------- |
| `toxic`         | Generally toxic content      |
| `severe_toxic`  | Severely toxic content       |
| `obscene`       | Obscene or vulgar language   |
| `threat`        | Threatening language         |
| `insult`        | Insulting content            |
| `identity_hate` | Hate speech targeting groups |

---

## Environment Variables

Create a `.env` file with the following variable:

| Variable         | Description                                     |
| ---------------- | ----------------------------------------------- |
| `ALLOWED_ORIGIN` | The allowed origin for CORS (e.g. frontend URL) |

---

## Getting Started

### Prerequisites

- Python >= 3.8
- pip
- Model weights file placed at `weight/phoBertModel_weights_50k_8_new.pth`

### Installation

```bash
pip install -r requirements.txt
```

### Development

```bash
python app.py
```

The server will start at `http://localhost:5000`.

### Production

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## Goals

- Gained a deeper understanding of BERT and PhoBERT — including their architecture, parameters, and how the model processes and represents Vietnamese text
- Learned how to integrate a pre-trained Vietnamese NLP model (PhoBERT) into a production-ready REST API
- Built a multi-label text classification pipeline using PyTorch and HuggingFace Transformers
- Fine-tuned PhoBERT on a Vietnamese comment dataset with 6 toxicity labels
- Designed a lightweight Flask service with CORS support for integration with the main backend
- Gained practical experience with deep learning model loading, inference optimization, and threshold-based decision making
