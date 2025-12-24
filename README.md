# PII NER - Named Entity Recognition for Personally Identifiable Information

A token-level Named Entity Recognition (NER) model for detecting and tagging Personally Identifiable Information (PII) in Speech-to-Text (STT) style transcripts. Built with HuggingFace Transformers and PyTorch.

## 🎯 Overview

This project trains a transformer-based NER model to identify sensitive PII entities in conversational text, particularly designed for Indian English transcripts from voice/call data. The model handles various formats including spoken numbers, informal email/phone representations, and code-mixed text.

## 📋 Supported Entity Types

| Entity | Description | Example |
|--------|-------------|---------|
| `PERSON_NAME` | Full names of individuals | "aditya rao", "neha dubey" |
| `PHONE` | Phone numbers (digits or spoken) | "79270 65379", "seven nine six three eight..." |
| `EMAIL` | Email addresses (standard or spoken) | "aditya.rao@gmail.com", "aditya dot rao at gmail dot com" |
| `DATE` | Dates in various formats | "18/05/2024", "5 july 2023", "14th november" |
| `CITY` | City names | "hyderabad", "mumbai", "pune" |
| `LOCATION` | Specific locations/areas | "electronic city", "koramangala", "mg road" |
| `CREDIT_CARD` | Credit card numbers | "3518439055295806", "three four eight eight..." |

## 🗂️ Project Structure

```
├── data/
│   ├── train.jsonl          # Training data
│   ├── dev.jsonl            # Development/validation data
│   ├── test.jsonl           # Test data
│   └── stress.jsonl         # Stress test (code-mixed, noisy inputs)
├── src/
│   ├── train.py             # Model training script
│   ├── predict.py           # Inference/prediction script
│   ├── eval_span_f1.py      # Span-level F1 evaluation
│   ├── measure_latency.py   # Latency benchmarking
│   ├── dataset.py           # PyTorch Dataset for NER
│   ├── model.py             # Model creation utilities
│   └── labels.py            # Label definitions (BIO tagging)
├── scripts/
│   └── synthesize_data.py   # Data augmentation/synthesis
├── out/                     # Model checkpoints and predictions
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- torch
- transformers
- numpy
- tqdm
- seqeval

### 2. Training

Train a DistilBERT-based NER model:

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 5 \
  --batch_size 8 \
  --lr 4e-5
```

**Training Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `distilbert-base-uncased` | Pretrained model from HuggingFace |
| `--train` | `data/train.jsonl` | Training data path |
| `--dev` | `data/dev.jsonl` | Validation data path |
| `--out_dir` | `out` | Output directory for model |
| `--batch_size` | `8` | Training batch size |
| `--epochs` | `5` | Number of training epochs |
| `--lr` | `4e-5` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay for AdamW |
| `--max_length` | `256` | Maximum sequence length |
| `--warmup_ratio` | `0.1` | Warmup steps ratio |
| `--max_grad_norm` | `1.0` | Gradient clipping norm |
| `--device` | `cuda`/`cpu` | Device (auto-detected) |

### 3. Prediction

Run inference on new data:

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

**Output Format:**
```json
{
  "utt_0048": [
    {"start": 8, "end": 18, "label": "PERSON_NAME"},
    {"start": 24, "end": 33, "label": "CITY"},
    {"start": 46, "end": 57, "label": "PHONE"}
  ]
}
```

### 4. Evaluation

Compute span-level Precision, Recall, and F1 scores:

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

**Sample Output:**
```
Per-entity metrics:
CITY            P=0.950 R=0.920 F1=0.935
CREDIT_CARD     P=0.880 R=0.910 F1=0.895
DATE            P=0.970 R=0.950 F1=0.960
EMAIL           P=0.920 R=0.890 F1=0.905
PERSON_NAME     P=0.940 R=0.960 F1=0.950
PHONE           P=0.930 R=0.940 F1=0.935

Macro-F1: 0.930
```

### 5. Stress Testing

Evaluate on challenging inputs (code-mixed Hindi-English, spoken numbers):

```bash
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

### 6. Latency Measurement

Benchmark inference latency:

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

**Sample Output:**
```
Latency over 50 runs (batch_size=1):
  p50: 12.34 ms
  p95: 15.67 ms
```

## 📊 Data Format

**Input Format (JSONL):**
```json
{
  "id": "utt_0048",
  "text": "this is aditya rao from hyderabad my phone is 79270 65379",
  "entities": [
    {"start": 8, "end": 18, "label": "PERSON_NAME"},
    {"start": 24, "end": 33, "label": "CITY"},
    {"start": 46, "end": 57, "label": "PHONE"}
  ]
}
```

- `id`: Unique utterance identifier
- `text`: Raw transcript text (lowercase, STT-style)
- `entities`: List of entity spans with character-level offsets
  - `start`: Start character index (inclusive)
  - `end`: End character index (exclusive)
  - `label`: Entity type

## 🔧 Data Synthesis

Generate additional synthetic training data:

```bash
python scripts/synthesize_data.py \
  --train_in data/train.jsonl \
  --dev_in data/dev.jsonl \
  --train_count 100 \
  --dev_count 20
```

The synthesizer creates realistic variations including:
- Spoken phone numbers ("seven nine six three...")
- Spoken credit card numbers with various formats
- Email addresses in spoken format ("name dot surname at gmail dot com")
- Multiple date formats (DD/MM/YYYY, "14th november", "5 july 2023")

## 🏗️ Model Architecture

- **Base Model:** DistilBERT (or any HuggingFace transformer)
- **Task:** Token Classification with BIO tagging scheme
- **Labels:** 15 labels (O + B/I for 7 entity types)
- **Optimizer:** AdamW with linear warmup scheduler
- **Loss:** Cross-entropy with label masking

## 📝 BIO Tagging Scheme

The model uses standard BIO (Beginning-Inside-Outside) tagging:
- `B-{ENTITY}`: Beginning of an entity
- `I-{ENTITY}`: Inside/continuation of an entity
- `O`: Outside any entity

## 🎛️ Post-Processing

The prediction script includes intelligent post-processing:
- **Phone validation:** Requires ≥8 digits or ≥5 spoken words
- **Credit card validation:** Requires ≥12 digits or ≥10 spoken words
- **Email validation:** Checks for "at"/"dot" patterns or @ symbol
- **Date validation:** Requires digits or month words

## 📈 Metrics

Evaluation computes:
- **Per-entity Precision, Recall, F1** (exact span match)
- **Macro-F1** across all entity types
- **PII-specific F1** (CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE)

## 🔮 Future Improvements

- [ ] Add confidence scores to predictions
- [ ] Support for batch inference
- [ ] ONNX export for production deployment
- [ ] Active learning for data annotation
- [ ] Multi-language support
