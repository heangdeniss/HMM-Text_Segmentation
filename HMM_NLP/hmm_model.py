"""Reusable first-order HMM part-of-speech tagger.

The notebook (``Mode.ipynb``) trains the model from first principles.  This
module mirrors that pipeline so the trained parameters can be saved once,
reloaded without retraining, and used from scripts or the static website.

Commands
--------
    python hmm_model.py build          # train, evaluate, save pickle + web JSON
    python hmm_model.py tag "text"     # tag text with the saved model
    python hmm_model.py evaluate       # re-evaluate the saved model on the test split

Artifacts
---------
    ../models/hmm_pos_model.pkl        # compact Python model (counts, pi, A)
    ../docs/model/hmm_pos_model.json   # sparse model consumed by the website
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
DEFAULT_ALPHA = 1e-9
EPSILON = 1e-12
OOV = "<OOV>"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "hmm_pos_model.pkl"
DEFAULT_WEB_PATH = REPO_ROOT / "docs" / "model" / "hmm_pos_model.json"

TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*|[^\s\w]", re.UNICODE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_splits(seed: int = SEED, test_ratio: float = 0.2):
    """Download (if needed) the Brown corpus and reproduce the notebook split."""
    import nltk
    from nltk.corpus import brown

    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)

    sentences = list(brown.tagged_sents(tagset="universal"))
    random.seed(seed)
    random.shuffle(sentences)

    split_index = int((1.0 - test_ratio) * len(sentences))
    return sentences[:split_index], sentences[split_index:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_data, alpha: float = DEFAULT_ALPHA) -> dict:
    """Estimate pi, A and emission counts from tagged sentences."""
    initial_counts = Counter()
    transition_counts = defaultdict(Counter)
    emission_counts = defaultdict(Counter)
    tag_counts = Counter()
    vocabulary = set()
    tags = set()

    for sentence in train_data:
        previous_tag = None
        for index, (word, tag) in enumerate(sentence):
            word = word.lower()
            vocabulary.add(word)
            tags.add(tag)

            tag_counts[tag] += 1
            if index == 0:
                initial_counts[tag] += 1
            if previous_tag is not None:
                transition_counts[previous_tag][tag] += 1
            emission_counts[tag][word] += 1
            previous_tag = tag

    tags = sorted(tags)
    total_sentences = len(train_data)
    vocab_size = len(vocabulary)

    pi = {tag: initial_counts[tag] / total_sentences for tag in tags}

    transitions = {}
    for prev_tag in tags:
        total = sum(transition_counts[prev_tag].values())
        transitions[prev_tag] = {
            cur_tag: (transition_counts[prev_tag][cur_tag] / total if total else 0.0)
            for cur_tag in tags
        }

    model = {
        "pi": pi,
        "A": transitions,
        "emission_counts": {tag: dict(emission_counts[tag]) for tag in tags},
        "tag_counts": {tag: tag_counts[tag] for tag in tags},
        "tags": tags,
        "alpha": float(alpha),
        "vocab_size": vocab_size,
    }
    model["meta"] = {
        "name": "HMM Universal POS Tagger",
        "dataset": "Brown Corpus (universal tagset)",
        "seed": SEED,
        "alpha": float(alpha),
        "vocab_size": vocab_size,
        "num_tags": len(tags),
        "train_sentences": len(train_data),
        "train_tokens": int(sum(tag_counts.values())),
    }
    return model


def build_emissions(model: dict) -> dict:
    """Reconstruct the smoothed emission table B from stored counts."""
    alpha = model["alpha"]
    vocab_size = model["vocab_size"]
    emissions = {}
    for tag in model["tags"]:
        denominator = model["tag_counts"][tag] + alpha * (vocab_size + 1)
        emissions[tag] = {
            word: (count + alpha) / denominator
            for word, count in model["emission_counts"][tag].items()
        }
        emissions[tag][OOV] = alpha / denominator
    return emissions


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def log_viterbi(words, model: dict, emissions: dict | None = None):
    """Log-space Viterbi decoder (identical recursion to the notebook)."""
    if emissions is None:
        emissions = model.get("B") or build_emissions(model)

    tags = model["tags"]
    pi = model["pi"]
    transitions = model["A"]
    oov = {tag: emissions[tag][OOV] for tag in tags}

    if not words:
        return []

    scores = []
    backpointers = []

    # Initialisation
    scores.append({})
    backpointers.append({})
    first_word = words[0].lower()
    for tag in tags:
        emission_prob = emissions[tag].get(first_word, oov[tag])
        scores[0][tag] = math.log(pi.get(tag, EPSILON) + EPSILON) + math.log(
            emission_prob + EPSILON
        )
        backpointers[0][tag] = None

    # Forward pass
    for t in range(1, len(words)):
        scores.append({})
        backpointers.append({})
        word = words[t].lower()

        for cur_tag in tags:
            emission_prob = emissions[cur_tag].get(word, oov[cur_tag])
            best_score = float("-inf")
            best_prev = None

            for prev_tag in tags:
                transition_prob = transitions[prev_tag].get(cur_tag, 0.0)
                score = (
                    scores[t - 1][prev_tag]
                    + math.log(transition_prob + EPSILON)
                    + math.log(emission_prob + EPSILON)
                )
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag

            scores[t][cur_tag] = best_score
            backpointers[t][cur_tag] = best_prev

    # Termination + backtracking
    best_tag = max(scores[-1], key=scores[-1].get)
    best_tags = [best_tag]
    for t in range(len(words) - 1, 0, -1):
        best_tag = backpointers[t][best_tag]
        best_tags.insert(0, best_tag)

    return best_tags


def tag(words, model: dict, emissions: dict | None = None):
    return list(zip(words, log_viterbi(words, model, emissions)))


def tag_text(text: str, model: dict, emissions: dict | None = None):
    """Tokenize free text and tag each non-empty line independently."""
    results = []
    for line in text.splitlines():
        words = tokenize(line)
        if words:
            results.append(tag(words, model, emissions))
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: dict, test_data, emissions: dict | None = None) -> dict:
    if emissions is None:
        emissions = build_emissions(model)

    correct = total = 0
    for sentence in test_data:
        words = [word for word, _ in sentence]
        gold = [tag for _, tag in sentence]
        predicted = log_viterbi(words, model, emissions)
        for expected, actual in zip(gold, predicted):
            total += 1
            correct += int(expected == actual)

    return {
        "tokens": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_model(model: dict, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in model.items() if key != "B"}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    model["B"] = build_emissions(model)
    return model


def export_web_model(
    model: dict, path: Path | str = DEFAULT_WEB_PATH
) -> Path:
    """Write a sparse, language-agnostic model used by the browser demo.

    Only non-zero emission counts are stored.  Every missing (word, tag) pair
    receives exactly the same probability as ``<OOV>`` in the smoothed model,
    so the exported model is mathematically equivalent to the full table.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            **model.get("meta", {}),
            "epsilon": EPSILON,
            "oovToken": OOV,
        },
        "tags": model["tags"],
        "pi": model["pi"],
        "A": model["A"],
        "tagTotals": model["tag_counts"],
        "emissions": model["emission_counts"],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_build(args):
    train_data, test_data = load_splits()
    print(f"Train sentences: {len(train_data):,}")
    print(f"Test sentences : {len(test_data):,}")

    model = train(train_data, alpha=args.alpha)
    print(f"Tags ({len(model['tags'])}): {', '.join(model['tags'])}")
    print(f"Vocabulary: {model['vocab_size']:,}")

    if not args.no_eval:
        print("Evaluating on the held-out test split ...")
        result = evaluate(model, test_data)
        model["meta"].update(
            {
                "test_sentences": len(test_data),
                "test_tokens": result["tokens"],
                "correct_tokens": result["correct"],
                "accuracy": result["accuracy"],
            }
        )
        print(
            f"Accuracy: {result['accuracy'] * 100:.2f}% "
            f"({result['correct']:,}/{result['tokens']:,} tokens)"
        )

    model_path = save_model(model, args.model)
    web_path = export_web_model(model, args.web)
    print(f"Saved Python model : {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved web model    : {web_path} ({web_path.stat().st_size / 1e6:.2f} MB)")


def _cmd_tag(args):
    model = load_model(args.model)
    for line, pairs in zip(
        [line for line in args.text.splitlines() if line.strip()],
        tag_text(args.text, model),
    ):
        print(line)
        print("  " + " ".join(f"{word}/{tag}" for word, tag in pairs))


def _cmd_evaluate(args):
    model = load_model(args.model)
    _, test_data = load_splits(seed=model["meta"].get("seed", SEED))
    result = evaluate(model, test_data)
    print(f"Accuracy: {result['accuracy'] * 100:.2f}% ({result['correct']:,}/{result['tokens']:,})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Python model path")
    parser.add_argument("--web", default=str(DEFAULT_WEB_PATH), help="Web model path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="train and export the model")
    build.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    build.add_argument("--no-eval", action="store_true")
    build.set_defaults(func=_cmd_build)

    tag_parser = subparsers.add_parser("tag", help="tag text with a saved model")
    tag_parser.add_argument("text")
    tag_parser.set_defaults(func=_cmd_tag)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a saved model")
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
