# Submission Summary

## 1. Final Results

### Dev Set Metrics
- **Macro-F1**: 1.000
- **PII-only F1**: 1.000
- **Non-PII F1**: 1.000

**Per-entity Breakdown (Dev):**
- CITY: F1=1.000
- CREDIT_CARD: F1=1.000
- DATE: F1=1.000
- EMAIL: F1=1.000
- LOCATION: F1=1.000
- PERSON_NAME: F1=1.000
- PHONE: F1=1.000

### Stress Set Metrics
- **Macro-F1**: 0.569
- **PII-only F1**: 0.668
- **Non-PII F1**: 0.816

**Per-entity Breakdown (Stress):**
- CITY: F1=1.000
- CREDIT_CARD: F1=0.000
- DATE: F1=1.000
- EMAIL: F1=0.350
- PERSON_NAME: F1=0.398
- PHONE: F1=0.667

## 2. Code Base Explanation
The solution implements a token-level NER system using Hugging Face Transformers.
- **`src/dataset.py`**: Handles data loading and conversion from JSONL to BIO format. It tokenizes text and aligns character-level labels to token-level labels.
- **`src/model.py`**: Defines the model architecture using `AutoModelForTokenClassification`.
- **`src/train.py`**: Training loop with PyTorch, using AdamW optimizer and linear scheduler.
- **`src/predict.py`**: Inference script that converts model predictions back to character-level spans, with post-processing rules to filter false positives.
- **`src/eval_span_f1.py`**: Evaluation script calculating Span-F1 and PII-specific metrics.

## 3. Model & Tokenizer
- **Model**: `distilbert-base-uncased`
- **Tokenizer**: `distilbert-base-uncased`
- **Reasoning**: Chosen for its balance between speed and performance, suitable for the latency constraints.

## 4. Key Hyperparameters
- **Epochs**: 3
- **Batch Size**: 8
- **Learning Rate**: 4e-5
- **Max Length**: 256
- **Optimizer**: AdamW

## 5. Latency Results
- **p50 Latency**: 34.65 ms
- **p95 Latency**: 52.76 ms
- **Hardware**: CPU (Local Environment)

**Trade-offs**:
The current latency (p95 ~55ms) exceeds the 20ms target. This is likely due to the use of `distilbert-base-uncased` which, while lighter than BERT, is still too heavy for strict <20ms CPU inference without further optimization (like quantization, ONNX export, or using an even smaller model like `prajjwal1/bert-mini`). The perfect F1 on Dev vs lower F1 on Stress suggests overfitting to the training distribution, which might be addressed with more diverse data augmentation or regularization.
