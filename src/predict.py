import argparse
import json
import os
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from labels import ID2LABEL, label_is_pii


def normalize_transition(label: str, current_label: str) -> str:
    if label == "O":
        return label
    prefix, ent_type = label.split("-", 1)
    if prefix == "I" and current_label != ent_type:
        return f"B-{ent_type}"
    return label


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        label = normalize_transition(label, current_label)
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


MONTH_WORDS = set(
    ["jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun", "june",
     "jul", "july", "aug", "august", "sep", "september", "oct", "october", "nov", "november",
     "dec", "december"]
)


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text)


def should_keep_span(span_text: str, label: str) -> bool:
    normalized = span_text.lower().strip()
    if not normalized:
        return False

    if label == "PHONE":
        return count_digits(normalized) >= 8 or len(normalized.split()) >= 5
    if label == "CREDIT_CARD":
        return count_digits(normalized) >= 12 or len(normalized.split()) >= 10
    if label == "EMAIL":
        return (" at " in normalized and " dot " in normalized) or bool(re.search(r"[A-Za-z0-9]+@[A-Za-z0-9]+", span_text))
    if label == "DATE":
        return count_digits(normalized) >= 2 or any(m in normalized for m in MONTH_WORDS)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            for s, e, lab in spans:
                if not should_keep_span(text[s:e], lab):
                    continue
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
