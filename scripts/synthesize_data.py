import argparse
import json
import os
import random
import string
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


FIRST_NAMES = [
    "aditya", "ananya", "arjun", "dhruv", "ishita", "karthik",
    "meera", "neha", "pallavi", "pooja", "pranav", "ramesh",
    "rohan", "sanjay", "sunita", "tushar", "vijay", "yash",
]

LAST_NAMES = [
    "agarwal", "banerjee", "bhatia", "chatterjee", "gupta", "iyer",
    "joshi", "krishnan", "mehta", "patel", "rao", "sharma",
    "singh", "verma",
]

CITIES = [
    "bengaluru", "chandigarh", "coimbatore", "delhi", "gurgaon", "hyderabad",
    "indore", "jaipur", "kochi", "kolkata", "lucknow", "mumbai",
    "nagpur", "pune", "surat", "trivandrum",
]

MICRO_LOCATIONS = [
    "banjara hills", "bellandur", "electronic city", "gachibowli",
    "hitech city", "indiranagar", "koramangala", "powai",
]

EMAIL_DOMAINS = [
    "gmail dot com",
    "gmail dot co dot in",
    "yahoo dot co dot in",
    "rediffmail dot com",
    "outlook dot com",
]

DATE_MONTH_WORDS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

NOISY_ZERO = ["oh", "zero"]


def random_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def random_city(rng: random.Random) -> str:
    return rng.choice(CITIES)


def random_micro_location(rng: random.Random) -> str:
    return rng.choice(MICRO_LOCATIONS)


def random_digits(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(length))


def digits_to_words(seq: str, rng: random.Random) -> str:
    out = []
    for d in seq:
        if d == "0":
            out.append(rng.choice(NOISY_ZERO))
        else:
            out.append(DIGIT_WORDS[d])
    return " ".join(out)


def format_phone(rng: random.Random) -> Tuple[str, str]:
    digits = random_digits(rng, 10)
    style = rng.choice(["spaced", "chunked", "spoken"])
    if style == "spoken":
        return digits, digits_to_words(digits, rng)
    if style == "chunked":
        return digits, f"{digits[:5]} {digits[5:]}"
    return digits, " ".join(digits)


def format_credit_card(rng: random.Random) -> Tuple[str, str]:
    digits = random_digits(rng, 16)
    style = rng.choice(["spoken", "chunks", "pairs"])
    if style == "spoken":
        repr_text = digits_to_words(digits, rng)
    elif style == "pairs":
        repr_text = " ".join("".join(pair) for pair in zip(digits[::2], digits[1::2]))
    else:
        repr_text = " ".join([digits[i : i + 4] for i in range(0, 16, 4)])
    return digits, repr_text


def format_email(name: str, rng: random.Random) -> str:
    name_part = name.replace(" ", " dot ")
    if rng.random() < 0.3:
        name_part = name_part.replace("a", "aa")
    return f"{name_part} at {rng.choice(EMAIL_DOMAINS)}"


def format_date(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    year = rng.randint(2023, 2028)
    style = rng.choice(["slash", "dash", "words"])
    if style == "slash":
        month = rng.randint(1, 12)
        return f"{day:02d}/{month:02d}/{year}"
    if style == "dash":
        month = rng.randint(1, 12)
        return f"{day:02d}-{month:02d}-{year}"
    month_word = rng.choice(DATE_MONTH_WORDS)
    return f"{day} {month_word} {year}"


class TranscriptBuilder:
    def __init__(self):
        self._parts: List[str] = []
        self._len = 0
        self.entities: List[Dict[str, object]] = []

    def append(self, text: str, label: str = None):
        if not text:
            return
        if self._parts and not self._parts[-1].endswith(" ") and not text.startswith(" "):
            self._parts.append(" ")
            self._len += 1
        start = self._len
        self._parts.append(text)
        self._len += len(text)
        end = self._len
        if label:
            self.entities.append({"start": start, "end": end, "label": label})

    def text(self) -> str:
        return "".join(self._parts).strip()


TemplateFn = Callable[[random.Random, str], Dict[str, object]]


def template_phone_email(rng: random.Random, uid: str) -> Dict[str, object]:
    tb = TranscriptBuilder()
    name = random_name(rng)
    city = random_city(rng)
    phone_digits, phone_repr = format_phone(rng)
    email_repr = format_email(name, rng)
    date_repr = format_date(rng)

    tb.append("this is")
    tb.append(name, "PERSON_NAME")
    tb.append("calling from")
    tb.append(city, "CITY")
    tb.append("my phone number is")
    tb.append(phone_repr, "PHONE")
    tb.append("please email me at")
    tb.append(email_repr, "EMAIL")
    tb.append("we can meet on")
    tb.append(date_repr, "DATE")

    return {"id": uid, "text": tb.text(), "entities": tb.entities}


def template_travel(rng: random.Random, uid: str) -> Dict[str, object]:
    tb = TranscriptBuilder()
    name = random_name(rng)
    city = random_city(rng)
    date_repr = format_date(rng)

    tb.append("i am")
    tb.append(name, "PERSON_NAME")
    tb.append("travelling to")
    tb.append(city, "CITY")
    tb.append("on")
    tb.append(date_repr, "DATE")

    return {"id": uid, "text": tb.text(), "entities": tb.entities}


def template_location(rng: random.Random, uid: str) -> Dict[str, object]:
    tb = TranscriptBuilder()
    loc = random_micro_location(rng)
    city = random_city(rng)

    tb.append("the office is near")
    tb.append(loc, "LOCATION")
    tb.append("in")
    tb.append(city, "CITY")
    tb.append("today")

    return {"id": uid, "text": tb.text(), "entities": tb.entities}


def template_credit_card(rng: random.Random, uid: str) -> Dict[str, object]:
    tb = TranscriptBuilder()
    name = random_name(rng)
    city = random_city(rng)
    cc_digits, cc_repr = format_credit_card(rng)
    exp_date = format_date(rng)
    email_repr = format_email(name, rng)

    tb.append("my name is")
    tb.append(name, "PERSON_NAME")
    tb.append("i live in")
    tb.append(city, "CITY")
    tb.append("my credit card number is")
    tb.append(cc_repr, "CREDIT_CARD")
    tb.append("it expires on")
    tb.append(exp_date, "DATE")
    tb.append("you can email me on")
    tb.append(email_repr, "EMAIL")

    return {"id": uid, "text": tb.text(), "entities": tb.entities}


def template_phone_only(rng: random.Random, uid: str) -> Dict[str, object]:
    tb = TranscriptBuilder()
    name = random_name(rng)
    phone_digits, phone_repr = format_phone(rng)

    tb.append("this is")
    tb.append(name, "PERSON_NAME")
    tb.append("my phone is")
    tb.append(phone_repr, "PHONE")
    tb.append("please call me tomorrow")

    return {"id": uid, "text": tb.text(), "entities": tb.entities}


TEMPLATES: Sequence[TemplateFn] = [
    template_phone_email,
    template_travel,
    template_location,
    template_credit_card,
    template_phone_only,
]


def load_jsonl(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, items: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def generate_examples(rng: random.Random, count: int, prefix: str, existing_ids: set) -> List[Dict[str, object]]:
    examples = []
    for idx in range(count):
        uid = f"{prefix}_{idx:04d}"
        while uid in existing_ids:
            idx += 1
            uid = f"{prefix}_{idx:04d}"
        template = rng.choice(TEMPLATES)
        ex = template(rng, uid)
        examples.append(ex)
        existing_ids.add(uid)
    return examples


@dataclass
class Args:
    train_path: str
    dev_path: str
    train_extra: int
    dev_extra: int
    train_out: str
    dev_out: str
    seed: int
    backup: bool


def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/train.jsonl", help="Existing train jsonl to augment")
    ap.add_argument("--dev_path", default="data/dev.jsonl", help="Existing dev jsonl to augment")
    ap.add_argument("--train_extra", type=int, default=400, help="Number of synthetic train examples to add")
    ap.add_argument("--dev_extra", type=int, default=120, help="Number of synthetic dev examples to add")
    ap.add_argument("--train_out", default=None, help="Output path for augmented train (default: overwrite train_path)")
    ap.add_argument("--dev_out", default=None, help="Output path for augmented dev (default: overwrite dev_path)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--backup", action="store_true", help="Store backup *.bak before overwriting")
    parsed = ap.parse_args()
    return Args(
        train_path=parsed.train_path,
        dev_path=parsed.dev_path,
        train_extra=parsed.train_extra,
        dev_extra=parsed.dev_extra,
        train_out=parsed.train_out or parsed.train_path,
        dev_out=parsed.dev_out or parsed.dev_path,
        seed=parsed.seed,
        backup=parsed.backup,
    )


def maybe_backup(path: str) -> None:
    if not os.path.exists(path):
        return
    backup_path = path + ".bak"
    if os.path.exists(backup_path):
        return
    with open(path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    train_items = load_jsonl(args.train_path)
    dev_items = load_jsonl(args.dev_path)

    existing_ids = {item["id"] for item in train_items + dev_items}

    if args.train_extra > 0:
        train_items.extend(generate_examples(rng, args.train_extra, "synth_train", existing_ids))
    if args.dev_extra > 0:
        dev_items.extend(generate_examples(rng, args.dev_extra, "synth_dev", existing_ids))

    if args.backup:
        maybe_backup(args.train_out)
        maybe_backup(args.dev_out)

    if args.train_out:
        write_jsonl(args.train_out, train_items)
        print(f"Wrote augmented train set with {len(train_items)} examples -> {args.train_out}")

    if args.dev_out:
        write_jsonl(args.dev_out, dev_items)
        print(f"Wrote augmented dev set with {len(dev_items)} examples -> {args.dev_out}")


if __name__ == "__main__":
    main()

