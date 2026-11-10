# COMP4153 Natural Language Processing — Lab 5
# Information Extraction and Structured NLP
# Student version | VS Code / Python 3.11+ | CPU or CUDA
#
# Setup (VS Code terminal):
#   python -m pip install numpy torch
# Run:
#   python COMP4153_NLP_Lab5_Student.py
#
# Use the VS Code Python extension for debugging (F5, breakpoints,
# variable inspection). Optional # %% cells can run interactively with
# the Jupyter extension; execute in order to initialise the models.
# Training data and models are deliberately small teaching examples.

from __future__ import annotations


# %%
# =====================================================================
# PART A — GUIDED EXAMPLES
# =====================================================================
# Information Extraction and Structured NLP (based on Lecture 5).
# Topics: BIO tagging, span extraction, entity linking, relations, events,
# structured records, validation and evaluation.

# %%
# Conventions
# Entity types: ORG, PER, PROJ, EVENT and DATE.
# BIO tags mark a beginning (B), continuation (I) or non-entity (O).
# Entity spans use zero-based, inclusive token positions.

# %%
# A0.1 — Imports, reproducibility and CPU/GPU selection
# Choose CPU or CUDA and fix random seeds for repeatable toy experiments.

import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn

# Entity outputs: (mention, type, first token, last token).
EntitySpan = tuple[str, str, int, int]
# Span candidates: (first token, last token, mention).
CandidateSpan = tuple[int, int, str]

SEED = 7

def set_seed(seed=SEED):
    """Reset the random generators used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("Device: CUDA GPU")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")

    available_threads = os.cpu_count() or 1
    torch.set_num_threads(min(4, available_threads))
    print("PyTorch CPU threads:", torch.get_num_threads())

print("PyTorch version:", torch.__version__)

# %%
# A1 — BIO entity extraction
# Recognise both the boundaries and types of entity mentions.
# B-ORG I-ORG identifies one two-token organisation mention.

# %%
# A1.1 — Lecture 5 running example and BIO encoding
# BIO assigns one tag to each word in a pre-tokenised sentence.

walk_tokens = [
    "Nova", "Robotics", ",", "a", "student", "team", "from", "UNNC", ",",
    "will", "demonstrate", "a", "solar-powered", "rover", "at", "the",
    "Ningbo", "Campus", "Innovation", "Showcase", "on", "18", "May", "."
]

walk_labels = [
    "B-ORG", "I-ORG", "O", "O", "O", "O", "O", "B-ORG", "O",
    "O", "O", "O", "B-PROJ", "I-PROJ", "O", "O",
    "B-EVENT", "I-EVENT", "I-EVENT", "I-EVENT", "O",
    "B-DATE", "I-DATE", "O"
]

print(f"{'TOKEN':22s} {'BIO LABEL':12s}")
print("-" * 36)

for token, label in zip(walk_tokens, walk_labels):
    print(f"{token:22s} {label:12s}")

# %%
# BIO decoder
# Convert predicted token tags into complete spans before evaluation.

# %%
# A1.2 — BIO decoder
# An unexpected I-TYPE starts a span so malformed tags remain inspectable.

def decode_bio(tokens: list[str], labels: list[str]) -> list[EntitySpan]:
    """Convert BIO labels into entity spans (text, type, start, end).

    Args:
        tokens: The words in their original order.
        labels: One BIO tag per word, for example "B-ORG" or "O".

    Returns:
        Extracted entities with inclusive, zero-based start/end indices.

    Example:
        >>> decode_bio(["Nova", "Robotics", "arrived"], ["B-ORG", "I-ORG", "O"])
        [('Nova Robotics', 'ORG', 0, 1)]
    """
    entities = []
    i = 0

    while i < len(tokens):
        label = labels[i]

        if label == "O":
            i += 1
            continue  # Skip this token and return to the top of the loop.

        if "-" not in label:
            i += 1
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix not in {"B", "I"}:
            i += 1
            continue

        start = i
        i += 1

        # Continue an entity only while the BIO type matches.
        while i < len(tokens) and labels[i] == f"I-{entity_type}":
            i += 1

        end = i - 1
        text = " ".join(tokens[start:end + 1])
        entities.append((text, entity_type, start, end))

    return entities


walk_entities = decode_bio(walk_tokens, walk_labels)
print("Decoded spans:", walk_entities)

# %%
# A1.3 — Small visualisation helper
# Display spans directly in the terminal (no notebook HTML needed).

def render_entities(tokens: list[str], entities: list[tuple[str, str, int, int]]) -> None:
    """Show entities in ordinary terminal text, e.g. [Nova Robotics: ORG].

    This replaces script-only HTML output so it works in VS Code's terminal.
    """
    start_map = {start: (text, typ, end) for text, typ, start, end in entities}
    pieces = []
    i = 0
    while i < len(tokens):
        if i in start_map:
            text, entity_type, end = start_map[i]
            pieces.append(f"[{text}: {entity_type}]")
            i = end + 1  # Advance past all words belonging to this entity.
        else:
            pieces.append(tokens[i])
            i += 1
    print(" ".join(pieces))

render_entities(walk_tokens, walk_entities)

# %%
# A1.1 — BiLSTM token classifier
# Embeddings -> bidirectional LSTM -> token-level BIO classification.
# This small model is instructional: it is not a production-quality extractor.

# %%
# A1.4 — Tiny teaching dataset
# The tiny labelled dataset is deliberately too small for general use.

token_examples = [
    (
        ["Nova", "Robotics", "will", "demonstrate", "a", "solar-powered", "rover",
         "at", "the", "Ningbo", "Campus", "Innovation", "Showcase", "on", "18", "May"],
        ["B-ORG", "I-ORG", "O", "O", "O", "B-PROJ", "I-PROJ",
         "O", "O", "B-EVENT", "I-EVENT", "I-EVENT", "I-EVENT", "O", "B-DATE", "I-DATE"]
    ),

    (
        ["Nova","Robotics","will","demonstrate","a","solar-powered","rover","at",
         "Innovation","Showcase","on","18","May"],
        ["B-ORG","I-ORG","O","O","O","B-PROJ","I-PROJ","O",
         "B-EVENT","I-EVENT","O","B-DATE","I-DATE"]
    ),
    (
        ["Lin","Mei","leads","Nova","Robotics"],
        ["B-PER","I-PER","O","B-ORG","I-ORG"]
    ),
    (
        ["Aero","Lab","built","a","delivery","drone","for","Tech","Expo"],
        ["B-ORG","I-ORG","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT"]
    ),
    (
        ["Maya","Chen","presented","the","vision","system","at","Research","Day"],
        ["B-PER","I-PER","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT"]
    ),
    (
        ["Green","Mobility","Team","will","show","an","electric","kart","at",
         "Innovation","Fair","on","21","June"],
        ["B-ORG","I-ORG","I-ORG","O","O","O","B-PROJ","I-PROJ","O",
         "B-EVENT","I-EVENT","O","B-DATE","I-DATE"]
    ),
    (
        ["Noah","Wang","joined","Quantum","Works"],
        ["B-PER","I-PER","O","B-ORG","I-ORG"]
    ),
    (
        ["Robotics","Club","tested","the","inspection","robot","at","Maker","Festival"],
        ["B-ORG","I-ORG","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT"]
    ),
    (
        ["Sofia","Li","designed","a","medical","chatbot","for","Student","Showcase"],
        ["B-PER","I-PER","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT"]
    ),
    (
        ["Ocean","AI","will","launch","Reef","Watch","at","Blue","Tech","Expo","on","3","July"],
        ["B-ORG","I-ORG","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT","I-EVENT",
         "O","B-DATE","I-DATE"]
    ),
    (
        ["Ethan","Zhou","works","with","Smart","Farming","Lab"],
        ["B-PER","I-PER","O","O","B-ORG","I-ORG","I-ORG"]
    ),
    (
        ["Data","Science","Society","demoed","a","campus","assistant","at","Open","Day"],
        ["B-ORG","I-ORG","I-ORG","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT"]
    ),
    (
        ["Iris","Sun","introduced","the","path","planner","at","Demo","Day","on","12","April"],
        ["B-PER","I-PER","O","O","B-PROJ","I-PROJ","O","B-EVENT","I-EVENT",
         "O","B-DATE","I-DATE"]
    ),
]

LABELS = [
    "O",
    "B-ORG", "I-ORG",
    "B-PER", "I-PER",
    "B-PROJ", "I-PROJ",
    "B-EVENT", "I-EVENT",
    "B-DATE", "I-DATE",
]

label_to_id = {label: i for i, label in enumerate(LABELS)}
id_to_label = {i: label for label, i in label_to_id.items()}

print("Examples:", len(token_examples))
print("BIO labels:", LABELS)

# %%
# A1.5 — Vocabulary and batching
# <PAD> fills shorter sequences; <UNK> represents unseen vocabulary.

SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
vocab = SPECIAL_TOKENS + sorted({
    token
    for sentence, _ in token_examples
    for token in sentence
})

token_to_id = {token: i for i, token in enumerate(vocab)}
PAD_ID = token_to_id["<PAD>"]
UNK_ID = token_to_id["<UNK>"]

MAX_LEN = max(len(sentence) for sentence, _ in token_examples)

def make_token_batch(examples):
    input_ids, label_ids, attention_masks = [], [], []

    for tokens, labels in examples:
        ids = [token_to_id.get(tok, UNK_ID) for tok in tokens]
        y = [label_to_id[label] for label in labels]

        pad_n = MAX_LEN - len(ids)

        input_ids.append(ids + [PAD_ID] * pad_n)

        # The -100 label is ignored by CrossEntropyLoss.
        label_ids.append(y + [-100] * pad_n)  # ignored by CrossEntropyLoss

        # The mask distinguishes real tokens from padding.
        attention_masks.append([1] * len(ids) + [0] * pad_n)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(label_ids, dtype=torch.long, device=device),
        torch.tensor(attention_masks, dtype=torch.bool, device=device),
    )

X_tok, Y_tok, M_tok = make_token_batch(token_examples)

print("Vocabulary size:", len(vocab))
print("Batch shape:", X_tok.shape)

# %%
# A1.6 — BiLSTM token classifier
# The bidirectional LSTM combines left and right sentence context.

class TinyTokenClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim=48, hidden_size=40):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers=1,
                               bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        embeddings = self.token_embedding(input_ids)

        # Pack sequences so padding does not affect BiLSTM states.
        lengths = attention_mask.long().sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.encoder(packed)
        contextual, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=input_ids.size(1))

        # Produce BIO-label scores for every token.
        logits = self.classifier(contextual)
        return (logits, contextual) if return_hidden else logits


token_model = TinyTokenClassifier(vocab_size=len(vocab), num_labels=len(LABELS)).to(device)
print(token_model)

# %%
# Training
# Cross-entropy learns gold BIO labels; padding is excluded from the loss.

# %%
# A1.7 — Train the token classifier
# Ignore padded labels (-100), backpropagate, and update weights with Adam.

set_seed()

optimizer = torch.optim.Adam(
    token_model.parameters(),
    lr=0.01,
)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

history = []
token_train_start = time.perf_counter()

for epoch in range(121):
    token_model.train()

    optimizer.zero_grad()

    logits = token_model(X_tok, M_tok)

    # Flatten batch and time dimensions before cross-entropy.
    loss = criterion(
        logits.reshape(-1, len(LABELS)),
        Y_tok.reshape(-1),
    )

    # Compute gradients and update the network.
    loss.backward()
    torch.nn.utils.clip_grad_norm_(token_model.parameters(), max_norm=1.0)
    optimizer.step()

    history.append(float(loss.detach().cpu()))

    if epoch % 30 == 0 or epoch == 120:
        print(f"epoch={epoch:3d} | loss={loss.item():.4f}")

token_train_seconds = time.perf_counter() - token_train_start

print(f"\nToken-classifier training time: {token_train_seconds:.2f} seconds")

# %%
# A1.8 — Inspect predictions
# Compare gold and predicted BIO labels, then decode whole entities.

token_model.eval()

with torch.no_grad():
    pred_ids = token_model(X_tok, M_tok).argmax(dim=-1)

example_index = 0
tokens, gold_labels = token_examples[example_index]
pred_labels = [
    id_to_label[int(i)]
    for i in pred_ids[example_index, :len(tokens)].cpu()
]

print(f"{'TOKEN':18s} {'GOLD':10s} {'PRED':10s}")
print("-" * 42)

for tok, gold, pred in zip(tokens, gold_labels, pred_labels):
    marker = "✓" if gold == pred else "✗"
    print(f"{tok:18s} {gold:10s} {pred:10s} {marker}")

pred_entities = decode_bio(tokens, pred_labels)

print("\nDecoded predicted entities:")
for e in pred_entities:
    print(e)

render_entities(tokens, pred_entities)

with torch.no_grad():
    train_logits = token_model(X_tok, M_tok)
    train_pred = train_logits.argmax(dim=-1)
    valid = Y_tok != -100
    train_token_accuracy = (
        (train_pred[valid] == Y_tok[valid]).float().mean().item()
    )

print(f"\nTraining-token accuracy: {train_token_accuracy:.3f}")

# %%
# A1.2 — Entity-level evaluation
# A correct prediction requires the exact start, end and entity type.
# Precision = correct / predicted; recall = correct / gold.

# %%
# A1.9 — Precision, recall and F1
# Count a true positive only when the full span and its type agree.

def prf1(
    predicted_items: list[EntitySpan] | set[EntitySpan],
    gold_items: list[EntitySpan] | set[EntitySpan],
) -> dict[str, float | int]:
    """Measure exact entity-level precision, recall, and F1.

    An entity is correct only when text, type, and both boundaries match.
    This function also returns the counts used to compute the metrics.
    """
    pred = set(predicted_items)
    gold = set(gold_items)

    # Set intersection counts exact matching entities.
    correct = len(pred & gold)

    precision = correct / len(pred) if pred else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "correct": correct,
        "predicted": len(pred),
        "gold": len(gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


gold_entities = decode_bio(tokens, gold_labels)
entity_scores = prf1(pred_entities, gold_entities)

print("Exact-match entity scores:", entity_scores)

# %%
# Exact-match error demonstration
# A boundary error or type error makes the entire entity incorrect.
# Training accuracy on these few examples does not measure generalisation.

# %%
# A1.10 — Deliberately introduce errors
# Boundary and type mistakes are both counted as mismatches.

gold_demo = [
    ("solar-powered rover", "PROJ", 5, 6),
    ("Innovation Showcase", "EVENT", 8, 9),
]

pred_demo = [
    ("rover", "PROJ", 6, 6),                 # boundary error
    ("Innovation Showcase", "ORG", 8, 9),    # type error
]

print(prf1(pred_demo, gold_demo))
print("\nNeither prediction counts as exactly correct.")

# %%
# A2 — Span-based extraction
# Enumerate contiguous candidates up to a maximum span length.
# Unlike one-label-per-token BIO, candidate spans can overlap.

# %%
# A2.1 — Generate candidate spans
# Candidate count grows as the maximum permitted span length increases.

def candidate_spans(tokens: list[str], max_span_length: int = 3) -> list[CandidateSpan]:
    """Generate contiguous candidate spans up to a maximum token length.

    No model is trained: this enumerates the spans a model *could* score.
    Start and end token indices are inclusive.
    """
    spans = []

    for start in range(len(tokens)):
        for end in range(
            start,
            min(len(tokens), start + max_span_length)
        ):
            spans.append(
                (start, end, " ".join(tokens[start:end + 1]))
            )

    return spans


span_sentence = ["Lin", "Mei", "leads", "the", "solar", "rover", "team"]
spans = candidate_spans(span_sentence, max_span_length=3)

print("Number of candidates:", len(spans))
print("First 12 candidates:")
for s in spans[:12]:
    print(s)

# %%
# Span representations
# A candidate may combine its left boundary, right boundary and pooled states.

# %%
# A2.2 — Construct one span representation from contextual hidden states
# Inspect contextual hidden states from the previously trained encoder.

def encode_tokens_with_token_model(tokens):
    ids = torch.tensor(
        [[token_to_id.get(tok, UNK_ID) for tok in tokens]],
        dtype=torch.long,
        device=device,
    )
    mask = torch.ones_like(ids, dtype=torch.bool)

    token_model.eval()
    with torch.no_grad():
        _, hidden = token_model(ids, mask, return_hidden=True)

    return hidden[0]


hidden = encode_tokens_with_token_model(walk_tokens)

i, j = 12, 13

span_vector = torch.cat([
    hidden[i],
    hidden[j],
    hidden[i:j + 1].mean(dim=0),
])

print("Hidden size:", hidden.shape[-1])
print("Span vector shape:", tuple(span_vector.shape))
print("Expected size = 3 × hidden size")

# %%
# Overlapping mentions
# For example, Ningbo may belong both to a university name and a location span.

# %%
# A2.3 — The two overlapping spans can coexist in a span representation
# Overlapping spans can coexist even when BIO tags cannot represent them.

nested_tokens = [
    "University", "of", "Nottingham", "Ningbo", "China"
]

nested_entities = [
    (0, 4, "ORG"),
    (3, 3, "LOCATION"),
]

for start, end, typ in nested_entities:
    print(
        f"{typ:8s}:",
        " ".join(nested_tokens[start:end + 1]),
        f"(tokens {start}..{end})"
    )

# %%
# A2.4 — Task-specific entities
# NER is one special case; an application can also extract projects or methods.

# %%
# A2.4 — Domain-specific spans from the Lecture 5 running example
# Project names and methods can be entities if the application needs them.

domain_tokens = [
    "The", "project", "lead", ",", "Lin", "Mei", ",", "said", "the", "rover",
    "uses", "a", "lightweight", "vision", "model", "to", "follow", "marked", "paths", "."
]

domain_entities = [
    (4, 5, "PER"),                  # Lin Mei
    (12, 14, "METHOD"),             # lightweight vision model
    (17, 18, "ENVIRONMENT_FEATURE") # marked paths
]

for start, end, entity_type in domain_entities:
    text = " ".join(domain_tokens[start:end + 1])
    print(
        f"{text:30s} -> {entity_type:20s} "
        f"(tokens {start}..{end})"
    )

print(
    "\nNotice: METHOD and ENVIRONMENT_FEATURE are domain-specific entity types, "
    "not standard NER categories."
)

# %%
# A3 — Entity linking
# Recognition: where? Typing: which category? Linking: which canonical identity?

# %%
# A3.1 — A tiny alias-based linker
# Resolve surface mentions to stable IDs using a small alias table.

knowledge_base = {
    "unnc_001": {
        "canonical_name": "University of Nottingham Ningbo China",
        "aliases": {
            "UNNC",
            "University of Nottingham Ningbo China",
        },
    },
    "nova_001": {
        "canonical_name": "Nova Robotics",
        "aliases": {"Nova Robotics"},
    },
}

def link_entity(mention, kb):
    matches = []

    for entity_id, entry in kb.items():
        if mention in entry["aliases"]:
            matches.append(
                (entity_id, entry["canonical_name"])
            )

    return matches


print("UNNC ->", link_entity("UNNC", knowledge_base))
print(
    "University of Nottingham Ningbo China ->",
    link_entity(
        "University of Nottingham Ningbo China",
        knowledge_base
    ),
)

# %%
# Entity-linking limitation
# Alias matching is only a toy linker; real cases may require context.

# %%
# A4 — Relation extraction
# Predict a relation for an ORDERED HEAD/TAIL pair.
# Marker normalisation focuses the model on context and argument direction.

# %%
# A4.1 — Relation training examples with entity markers
# Mark HEAD and TAIL explicitly; the same entity names can play different roles.

relation_examples = [
    (
        ["<HEAD>","Lin","Mei","</HEAD>","leads",
         "<TAIL>","Nova","Robotics","</TAIL>"],
        "leads"
    ),
    (
        ["<HEAD>","Maya","Chen","</HEAD>","leads",
         "<TAIL>","Aero","Lab","</TAIL>"],
        "leads"
    ),
    (
        ["<HEAD>","Ethan","Zhou","</HEAD>","leads",
         "<TAIL>","Smart","Farming","Lab","</TAIL>"],
        "leads"
    ),
    (
        ["<TAIL>","Lin","Mei","</TAIL>","leads",
         "<HEAD>","Nova","Robotics","</HEAD>"],
        "no_relation"
    ),
    (
        ["<TAIL>","Maya","Chen","</TAIL>","leads",
         "<HEAD>","Aero","Lab","</HEAD>"],
        "no_relation"
    ),
    (
        ["<HEAD>","Nova","Robotics","</HEAD>","built","a",
         "<TAIL>","solar-powered","rover","</TAIL>"],
        "built"
    ),
    (
        ["<HEAD>","Aero","Lab","</HEAD>","built","a",
         "<TAIL>","delivery","drone","</TAIL>"],
        "built"
    ),
    (
        ["<HEAD>","Robotics","Club","</HEAD>","built","the",
         "<TAIL>","inspection","robot","</TAIL>"],
        "built"
    ),
    (
        ["<TAIL>","Nova","Robotics","</TAIL>","built","a",
         "<HEAD>","solar-powered","rover","</HEAD>"],
        "no_relation"
    ),
    (
        ["<HEAD>","solar-powered","rover","</HEAD>","was","demonstrated","at",
         "<TAIL>","Innovation","Showcase","</TAIL>"],
        "demonstrated_at"
    ),
    (
        ["<HEAD>","electric","kart","</HEAD>","was","demonstrated","at",
         "<TAIL>","Innovation","Fair","</TAIL>"],
        "demonstrated_at"
    ),
    (
        ["<HEAD>","medical","chatbot","</HEAD>","was","demonstrated","at",
         "<TAIL>","Student","Showcase","</TAIL>"],
        "demonstrated_at"
    ),
    (
        ["<TAIL>","solar-powered","rover","</TAIL>","was","demonstrated","at",
         "<HEAD>","Innovation","Showcase","</HEAD>"],
        "no_relation"
    ),
    (
        ["<HEAD>","Lin","Mei","</HEAD>","visited",
         "<TAIL>","Aero","Lab","</TAIL>"],
        "no_relation"
    ),
    (
        ["<HEAD>","Nova","Robotics","</HEAD>","met",
         "<TAIL>","Ethan","Zhou","</TAIL>"],
        "no_relation"
    ),
]

REL_LABELS = [
    "leads",
    "built",
    "demonstrated_at",
    "no_relation",
]

rel_to_id = {r: i for i, r in enumerate(REL_LABELS)}
id_to_rel = {i: r for r, i in rel_to_id.items()}

print("Relation examples:", len(relation_examples))

# %%
# A4.2 — Relation vocabulary and batching
# Convert marked relation examples into fixed-length numeric tensors.

REL_SPECIAL = [
    "<PAD>",
    "<UNK>",
    "<CLS>",
    "<ENTITY>",
]

def normalise_marked_entities(tokens):
    normalised = []
    inside_entity = False
    entity_emitted = False

    for token in tokens:
        if token in {"<HEAD>", "<TAIL>"}:
            normalised.append(token)
            inside_entity = True
            entity_emitted = False
            continue  # Do not process the same marker as normal text.

        if token in {"</HEAD>", "</TAIL>"}:
            normalised.append(token)
            inside_entity = False
            entity_emitted = False
            continue

        if inside_entity:
            if not entity_emitted:
                normalised.append("<ENTITY>")
                entity_emitted = True
            continue

        normalised.append(token)

    return normalised


normalised_relation_examples = [
    (
        normalise_marked_entities(sentence),
        relation,
    )
    for sentence, relation in relation_examples
]

rel_vocab = REL_SPECIAL + sorted({
    token
    for sentence, _ in normalised_relation_examples
    for token in sentence
    if token not in REL_SPECIAL
})

rel_token_to_id = {
    token: i
    for i, token in enumerate(rel_vocab)
}

REL_PAD_ID = rel_token_to_id["<PAD>"]
REL_UNK_ID = rel_token_to_id["<UNK>"]
REL_CLS_ID = rel_token_to_id["<CLS>"]

REL_MAX_LEN = max(
    len(sentence) + 1
    for sentence, _ in normalised_relation_examples
)

def make_relation_batch(examples):
    input_ids, masks, y = [], [], []

    for tokens, relation in examples:
        tokens = normalise_marked_entities(tokens)

        ids = [REL_CLS_ID] + [
            rel_token_to_id.get(tok, REL_UNK_ID)
            for tok in tokens
        ]

        pad_n = REL_MAX_LEN - len(ids)

        input_ids.append(
            ids + [REL_PAD_ID] * pad_n
        )
        masks.append(
            [1] * len(ids) + [0] * pad_n
        )
        y.append(rel_to_id[relation])

    return (
        torch.tensor(
            input_ids,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            masks,
            dtype=torch.bool,
            device=device,
        ),
        torch.tensor(
            y,
            dtype=torch.long,
            device=device,
        ),
    )


X_rel, M_rel, Y_rel = make_relation_batch(
    relation_examples
)

print(
    "Example before normalisation:",
    relation_examples[0][0],
)

print(
    "Example after normalisation: ",
    normalise_marked_entities(
        relation_examples[0][0]
    ),
)

print(
    X_rel.shape,
    Y_rel.shape,
)

# %%
# A4.3 — BiLSTM relation classifier (ordered entity-pair representation)
# Predict one relation label for each ordered entity pair.

class TinyRelationClassifier(nn.Module):
    def __init__(self, vocab_size, num_relations, embedding_dim=48, hidden_size=40):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=REL_PAD_ID)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(6 * hidden_size, 64), nn.ReLU(), nn.Linear(64, num_relations))

    # Entity order is encoded by the marked-pair input.
    def forward(self, input_ids, attention_mask):
        emb = self.token_embedding(input_ids)
        lengths = attention_mask.long().sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.encoder(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden, batch_first=True, total_length=input_ids.size(1))
        head_marker = rel_token_to_id["<HEAD>"]
        tail_marker = rel_token_to_id["<TAIL>"]
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        head_pos = (input_ids == head_marker).int().argmax(dim=1) + 1
        tail_pos = (input_ids == tail_marker).int().argmax(dim=1) + 1
        head_repr = h[batch_indices, head_pos.clamp(max=h.size(1)-1)]
        tail_repr = h[batch_indices, tail_pos.clamp(max=h.size(1)-1)]
        context = (h * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).clamp(min=1).unsqueeze(-1)
        ordered_pair = torch.cat((head_repr, tail_repr, context), dim=-1)
        return self.classifier(ordered_pair)


relation_model = TinyRelationClassifier(vocab_size=len(rel_vocab), num_relations=len(REL_LABELS)).to(device)

# %%
# A4.4 — Train the relation classifier
# Train the relation model on the small marked-pair dataset.

set_seed()

rel_optimizer = torch.optim.Adam(
    relation_model.parameters(),
    lr=0.01,
)
rel_criterion = nn.CrossEntropyLoss()

relation_train_start = time.perf_counter()

for epoch in range(151):
    relation_model.train()
    rel_optimizer.zero_grad()

    logits = relation_model(X_rel, M_rel)
    loss = rel_criterion(logits, Y_rel)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(relation_model.parameters(), max_norm=1.0)
    rel_optimizer.step()

    if epoch % 50 == 0 or epoch == 150:
        print(f"epoch={epoch:3d} | loss={loss.item():.4f}")

relation_train_seconds = time.perf_counter() - relation_train_start
print(f"\nRelation-classifier training time: {relation_train_seconds:.2f} seconds")

# %%
# A4.5 — Prediction helper
# Normalise entity surface words before predicting a new relation.

def predict_relation(marked_tokens):
    normalised_tokens = normalise_marked_entities(
        marked_tokens
    )

    ids = [REL_CLS_ID] + [
        rel_token_to_id.get(tok, REL_UNK_ID)
        for tok in normalised_tokens
    ]

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    mask = torch.ones_like(
        x,
        dtype=torch.bool,
    )

    relation_model.eval()

    with torch.no_grad():
        logits = relation_model(
            x,
            mask,
        )

        probs = torch.softmax(
            logits,
            dim=-1,
        )[0]

    best_id = int(
        probs.argmax().cpu()
    )

    probability_table = {
        id_to_rel[i]: float(
            probs[i].cpu()
        )
        for i in range(
            len(REL_LABELS)
        )
    }

    return (
        id_to_rel[best_id],
        probability_table,
    )

# %%
# A4.6 — Direction matters
# Swap HEAD and TAIL to inspect directional predictions.

normal_pair = [
    "<HEAD>", "Lin", "Mei", "</HEAD>",
    "leads",
    "<TAIL>", "Nova", "Robotics", "</TAIL>",
]

reversed_pair = [
    "<TAIL>", "Lin", "Mei", "</TAIL>",
    "leads",
    "<HEAD>", "Nova", "Robotics", "</HEAD>",
]

normal_pred, normal_probs = predict_relation(normal_pair)
reverse_pred, reverse_probs = predict_relation(reversed_pair)

print("HEAD=Lin Mei, TAIL=Nova Robotics")
print("prediction:", normal_pred)
print({k: round(v, 3) for k, v in normal_probs.items()})

print("\nHEAD=Nova Robotics, TAIL=Lin Mei")
print("prediction:", reverse_pred)
print({k: round(v, 3) for k, v in reverse_probs.items()})

# %%
# A4.1 — Pipeline error propagation
# If entity extraction misses a mention, relations involving it cannot be proposed.

# %%
# A4.1 — Simulate pipeline error propagation
# Only detected entities can be passed to a pipeline relation extractor.

def all_ordered_entity_pairs(entities):
    """Return every ordered pair of distinct extracted entity strings."""
    return [
        (head, tail)
        for head in entities
        for tail in entities
        if head != tail
    ]

gold_entities_pipeline = [
    "Lin Mei",
    "Nova Robotics",
    "solar-powered rover",
]

predicted_entities_pipeline = [
    "Lin Mei",
    "Nova Robotics",
]

gold_pairs = all_ordered_entity_pairs(gold_entities_pipeline)
predicted_pairs = all_ordered_entity_pairs(predicted_entities_pipeline)

target_pair = ("Nova Robotics", "solar-powered rover")

print("Target relation pair:", target_pair)
print("Available with gold entities?     ", target_pair in gold_pairs)
print("Available after entity miss?      ", target_pair in predicted_pairs)

print(
    "\nPipeline consequence: if a required entity is missing, "
    "some downstream relation candidates disappear completely."
)

# %%
# A5 — Event extraction
# A trigger identifies an occurrence; arguments fill roles such as Agent and Date.

# %%
# A5.1 — Event frame
# Group an event trigger and its role-labelled arguments into one structure.

event_frame = {
    "trigger": "demonstrate",
    "event_type": "Demonstration",
    "arguments": {
        "Agent": "Nova Robotics",
        "Artifact": "solar-powered rover",
        "Venue": "Ningbo Campus Innovation Showcase",
        "Date": "18 May",
    },
}

print(json.dumps(event_frame, indent=2))

# %%
# Multiple events
# Assign each argument to the correct event, not merely the nearest verb.

# %%
# A5.1 — Two event frames from one sentence
# Represent each occurrence separately when the sentence has multiple events.

multiple_event_frames = [
    {
        "trigger": "announced",
        "event_type": "Announcement",
        "arguments": {
            "Speaker": "Lin Mei",
            "Content": "demonstration plan",
            "Location": None,
            "Date": None,
        },
    },
    {
        "trigger": "demonstrate",
        "event_type": "Demonstration",
        "arguments": {
            "Agent": "Nova Robotics",
            "Artifact": "solar rover",
            "Location": "student innovation area",
            "Date": "18 May",
        },
    },
]

print(json.dumps(multiple_event_frames, indent=2))

print(
    "\nThe null values are meaningful: the source sentence does not specify "
    "a separate announcement location or announcement date."
)

# %%
# A5.2 — Structured record validation
# A parseable JSON candidate can still violate the schema or invent facts.

# %%
# A5.2 — Record schema and validator
# Check required fields, allowed types/categories and source evidence.

source_text_A = (
    "Nova Robotics, a student team from UNNC, will demonstrate "
    "a solar-powered rover at the Ningbo Campus Innovation Showcase on 18 May. "
    "The project lead, Lin Mei, said the rover uses a lightweight vision model "
    "to follow marked paths."
)

schema_A = {
    "team": {
        "type": str,
        "required": True,
        "nullable": False,
    },
    "project": {
        "type": str,
        "required": True,
        "nullable": False,
    },
    "event_type": {
        "type": str,
        "required": True,
        "nullable": False,
        "allowed": {"Demonstration"},
    },
    "venue": {
        "type": str,
        "required": True,
        "nullable": True,
    },
    "date": {
        "type": str,
        "required": True,
        "nullable": True,
    },
    "project_lead": {
        "type": str,
        "required": True,
        "nullable": True,
    },
}

candidate_record_A = {
    "team": "Nova Robotics",
    "project": "solar-powered rover",
    "event_type": "Demonstration",
    "venue": "Ningbo Campus Innovation Showcase",
    "date": "18 May",
    "project_lead": "Lin Mei",
}

EVENT_TRIGGER_EVIDENCE = {
    "Demonstration": {
        "demonstrate",
        "demonstrated",
        "demo",
        "demoed",
    }
}

# Validate structured outputs before downstream use.
def validate_record(record, source_text, schema):
    """Check schema constraints and simple source grounding.

    This teaching validator separates:
      1. schema validation — is the structure allowed?
      2. evidence validation — are extracted values supported by the source?
    """
    errors = []

    allowed_keys = set(schema)
    extra_keys = set(record) - allowed_keys

    if extra_keys:
        errors.append(
            f"schema error: unexpected fields {sorted(extra_keys)}"
        )

    for field, spec in schema.items():
        if field not in record:
            if spec.get("required", False):
                errors.append(
                    f"schema error: missing required field '{field}'"
                )
            continue

        value = record[field]

        if value is None:
            if not spec.get("nullable", False):
                errors.append(
                    f"schema error: field '{field}' may not be null"
                )
            continue

        if not isinstance(value, spec["type"]):
            errors.append(
                f"schema error: '{field}' has wrong type "
                f"{type(value).__name__}"
            )
            continue

        if "allowed" in spec and value not in spec["allowed"]:
            errors.append(
                f"schema error: '{field}' has disallowed value {value!r}"
            )

    source_lower = source_text.lower()

    for field in [
        "team",
        "project",
        "venue",
        "date",
        "project_lead",
    ]:
        value = record.get(field)

        if isinstance(value, str) and value.lower() not in source_lower:
            errors.append(
                f"evidence error: {field}={value!r} "
                "is not supported by a literal source span"
            )

    event_type = record.get("event_type")

    if isinstance(event_type, str):
        triggers = EVENT_TRIGGER_EVIDENCE.get(event_type, set())

        if triggers and not any(
            trigger in source_lower
            for trigger in triggers
        ):
            errors.append(
                f"evidence error: event_type={event_type!r} "
                "has no recognised trigger in the text"
            )

    return errors


print(json.dumps(candidate_record_A, indent=2))
print("\nValidation errors:")
print(
    validate_record(
        candidate_record_A,
        source_text_A,
        schema_A,
    )
)

# %%
# A5.2 — Inject two common failures
# Test the validator on plausible but incorrect candidate records.

bad_record_A = {
    "team": "Nova Robotics",
    "project": "solar-powered rover",
    "event_type": "Demonstration",
    "venue": "Ningbo Campus Innovation Showcase",
    "date": "18 May",
    "project_lead": "Daniel Wu",  # unsupported generation
    "organisation": "UNNC",        # not allowed by this schema
}

for error in validate_record(
    bad_record_A,
    source_text_A,
    schema_A,
):
    print("•", error)

# %%
# Missing information
# Use null for absent evidence instead of guessing a field value.

# %%
# A5.3 — Null is different from an invented value
# An absent date is not a licence to invent one.

source_without_date = (
    "Nova Robotics will demonstrate a solar-powered rover."
)

record_with_null = {
    "team": "Nova Robotics",
    "project": "solar-powered rover",
    "event_type": "Demonstration",
    "venue": None,
    "date": None,
    "project_lead": None,
}

record_with_invented_date = {
    "team": "Nova Robotics",
    "project": "solar-powered rover",
    "event_type": "Demonstration",
    "venue": None,
    "date": "18 May",       # unsupported by this source
    "project_lead": None,
}

print("Record using null:")
print(validate_record(
    record_with_null,
    source_without_date,
    schema_A,
))

print("\nRecord inventing a date:")
print(validate_record(
    record_with_invented_date,
    source_without_date,
    schema_A,
))

# %%
# Repeated records
# Use a list of team-project records to preserve which team built which project.

# %%
# A5.4 — Repeated records preserve relationships
# Keep related entities in the same record when multiple projects are mentioned.

bad_flat_record = {
    "teams": "Nova Robotics and GreenAI Club",
    "projects": "solar rover and energy-monitoring dashboard",
}

better_repeated_record = {
    "records": [
        {
            "team": "Nova Robotics",
            "project": "solar rover",
        },
        {
            "team": "GreenAI Club",
            "project": "energy-monitoring dashboard",
        },
    ]
}

print("Poor schema:")
print(json.dumps(bad_flat_record, indent=2))

print("\nBetter repeated-record schema:")
print(json.dumps(better_repeated_record, indent=2))

# %%
# Parse, validate, accept or correct
# Separately check JSON syntax, schema conformance and source grounding.

# %%
# A5.5 — Parse -> validate -> accept or retry
# Validation is a gate between candidate generation and storage.

candidate_json_text = """
{
  "team": "Nova Robotics",
  "project": "solar-powered rover",
  "event_type": "Demonstration",
  "venue": "Ningbo Campus Innovation Showcase",
  "date": "18 May",
  "project_lead": "Lin Mei"
}
"""

# A syntactically valid JSON object can still be ungrounded.
def parse_and_validate_candidate(json_text, source_text, schema):
    """A minimal version of a professional structured-extraction gate."""
    try:
        record = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, [f"JSON parse error: {exc.msg}"]

    if not isinstance(record, dict):
        return None, ["schema error: top-level JSON value must be an object"]

    errors = validate_record(record, source_text, schema)

    return record, errors


parsed_record, gate_errors = parse_and_validate_candidate(
    candidate_json_text,
    source_text_A,
    schema_A,
)

print("Parsed candidate:")
print(parsed_record)

print("\nGate result:")
if gate_errors:
    print("CORRECT RECORD BEFORE STORAGE")
    for error in gate_errors:
        print("•", error)
else:
    print("ACCEPT FOR STORAGE / REVIEW")

# %%
# A6 — Reasoning over relations
# Follow graph edges, e.g. person -> team -> project, to answer questions.

# %%
# A6.1 — Tiny graph reasoning without any graph library
# A relation graph supports multihop queries over extracted tuples.

relation_tuples_A = [
    ("Lin Mei", "leads", "Nova Robotics"),
    ("Nova Robotics", "built", "solar-powered rover"),
    ("solar-powered rover", "demonstrated_at", "Ningbo Campus Innovation Showcase"),
]

def who_leads_team_that_built(project, triples):
    builders = {
        head
        for head, relation, tail in triples
        if relation == "built" and tail == project
    }

    leaders = {
        head
        for head, relation, tail in triples
        if relation == "leads" and tail in builders
    }

    return sorted(leaders)


answer = who_leads_team_that_built(
    "solar-powered rover",
    relation_tuples_A,
)

print("Who leads the team that built the solar-powered rover?")
print(answer)

# %%
# A7 — Runtime check
# These tiny training tasks are intended to be feasible on a CPU.

# %%
# A7.1 — Core neural-training runtime
# Report training-time measurements rather than claiming production accuracy.

core_training_seconds = token_train_seconds + relation_train_seconds

print(f"Token classifier:    {token_train_seconds:7.2f} s")
print(f"Relation classifier: {relation_train_seconds:7.2f} s")
print("-" * 34)
print(f"Core neural training:{core_training_seconds:7.2f} s")

if core_training_seconds < 15 * 60:
    print("\n✓ Core training is below the 15-minute classroom target.")
else:
    print("\nWarning: runtime exceeded 15 minutes on this machine.")

# %%
# A8 — What counts as a correct extraction?
# Entities: exact span + type. Relations: head + label + tail + direction.
# Events: trigger/type + role arguments. Records: correct, grounded fields.

# %%
# =====================================================================
# PART B — EXERCISES
# =====================================================================
# Source: Harbour AI, a student team from UNNC, will demonstrate an
# autonomous litter-collecting boat at Green Campus Expo on 22 September.
# Project lead: Aisha Rahman. A lightweight vision model is mentioned.
# Target types: Harbour AI/UNNC=ORG; boat=PROJ; Expo=EVENT;
# 22 September=DATE; Aisha Rahman=PER.
# "lightweight vision model" is outside the target entity schema.

# %%
# B0.1 — Pretokenised exercise text
# Token indices are zero-based; use them for exact entity boundaries.

exercise_tokens = [
    "Harbour", "AI", "a", "student", "team", "from", "UNNC",
    "will", "demonstrate", "an",
    "autonomous", "litter-collecting", "boat",
    "at", "Green", "Campus", "Expo",
    "on", "22", "September",
    "The", "project", "lead", "Aisha", "Rahman",
    "said", "the", "boat", "uses", "a",
    "lightweight", "vision", "model"
]

print("Number of tokens:", len(exercise_tokens))
for i, tok in enumerate(exercise_tokens):
    print(f"{i:2d}: {tok}")

# %%
# B1 — Write BIO labels
# Assign one B-TYPE, I-TYPE or O label per exercise token.
# Use B-TYPE for one-token entities; keep the token and label counts equal.

# %%
# B1.1 — Exercise
# TODO: Replace None with the BIO labels for every token.

exercise_labels = None

# %%
# B1.2 — Self-check and visualise your answer
# Compare your decoded spans with the six gold entities.

gold_entities_B = [
    ("Harbour AI", "ORG", 0, 1),
    ("UNNC", "ORG", 6, 6),
    ("autonomous litter-collecting boat", "PROJ", 10, 12),
    ("Green Campus Expo", "EVENT", 14, 16),
    ("22 September", "DATE", 18, 19),
    ("Aisha Rahman", "PER", 23, 24),
]

if exercise_labels is None:
    print("B1 not completed yet.")
elif len(exercise_labels) != len(exercise_tokens):
    print(
        "Length mismatch:",
        len(exercise_labels),
        "labels for",
        len(exercise_tokens),
        "tokens"
    )
else:
    decoded_B = decode_bio(
        exercise_tokens,
        exercise_labels,
    )

    print("Decoded entities:")
    for entity in decoded_B:
        print(entity)

    print("\nExact-match score:")
    print(prf1(decoded_B, gold_entities_B))

    render_entities(exercise_tokens, decoded_B)

# %%
# B1 reflection
# Why is Harbour AI labelled B-ORG, I-ORG?
# What error results from labelling Green Campus Expo as B-EVENT, I-EVENT, O?

# %%
# B2 — Candidate span lengths
# Generate all spans with max_span_length=3, then with 2.
# Which of the six target entity mentions become impossible at length 2?
# Discuss candidate coverage versus computational cost.

# %%
# B2.1 — Exercise
# TODO: Generate span candidates with maximum lengths 3 and 2.

spans_L3 = None
spans_L2 = None


if spans_L3 is not None:
    print("Number of L=3 candidates:", len(spans_L3))

if spans_L2 is not None:
    print("Number of L=2 candidates:", len(spans_L2))

# %%
# B3 — Test the tagger on unseen text
# Decode model_labels_B; calculate exact-match precision, recall and F1.
# Diagnose two errors and suggest improvements.
# Unknown words map to <UNK>; generalisation will be limited.

# %%
# B3.1 — Inference helper (provided)
# The trained model reuses the original vocabulary and <UNK> token.

def predict_bio(tokens):
    ids = torch.tensor(
        [[token_to_id.get(tok, UNK_ID) for tok in tokens]],
        dtype=torch.long,
        device=device,
    )
    mask = torch.ones_like(ids, dtype=torch.bool)

    token_model.eval()

    with torch.no_grad():
        logits = token_model(ids, mask)
        pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

    return [id_to_label[i] for i in pred_ids]


model_labels_B = predict_bio(exercise_tokens)

for tok, pred in zip(exercise_tokens, model_labels_B):
    print(f"{tok:20s} {pred}")

# %%
# B3.2 — Exercise
# TODO: Decode predicted labels and evaluate the resulting entity spans.

model_entities_B = None
model_scores_B = None


if model_entities_B is not None:
    print("Predicted entities:")
    for item in model_entities_B:
        print(item)

    print("\nScores:", model_scores_B)
    render_entities(exercise_tokens, model_entities_B)

# %%
# B4 — Ordered relations with entity markers
# Text: Aisha Rahman leads Harbour AI. Harbour AI built the EcoSkimmer.
# EcoSkimmer was demonstrated at Green Campus Expo.
# Create marked pairs: (Aisha, Harbour AI), (Harbour AI, EcoSkimmer),
# (EcoSkimmer, Green Campus Expo), and reversed (Harbour AI, Aisha).
# Use <HEAD>...</HEAD> and <TAIL>...</TAIL>; run predict_relation().
# Why does reversing the pair matter? What if EcoSkimmer is missed?

# %%
# B4.1 — Exercise
# TODO: Supply four ordered, marked token lists, including a reversed pair.

marked_pair_1 = None
marked_pair_2 = None
marked_pair_3 = None
marked_pair_4 = None


for name, pair in [
    ("pair 1", marked_pair_1),
    ("pair 2", marked_pair_2),
    ("pair 3", marked_pair_3),
    ("pair 4", marked_pair_4),
]:
    if pair is None:
        print(name, "not completed")
    else:
        pred, probs = predict_relation(pair)
        print(
            name,
            "->",
            pred,
            {k: round(v, 3) for k, v in probs.items()}
        )

# %%
# B5 — Event frames and validated records
# Build a demonstration event: trigger, event_type and arguments
# (Agent, Artifact, Venue, Date).
# Build and validate a record: team, project, event_type, venue,
# date and project_lead. Use only facts supported by the text.
# For a shorter source without place/date/lead, explain why null is safer.

# %%
# B5.1 — Source and schema
# The deliberately bad example tests schema and evidence errors.

source_text_B = (
    "Harbour AI, a student team from UNNC, will demonstrate "
    "an autonomous litter-collecting boat at the Green Campus Expo "
    "on 22 September. The project lead, Aisha Rahman, said the boat "
    "uses a lightweight vision model."
)

schema_B = schema_A.copy()

bad_record_B = {
    "team": "Harbour AI",
    "project": "autonomous litter-collecting boat",
    "event_type": "Demo",       # schema problem
    "venue": "Green Campus Expo",
    "date": "22 September",
    "project_lead": "Daniel Wu",  # unsupported generation
}

for error in validate_record(
    bad_record_B,
    source_text_B,
    schema_B,
):
    print("•", error)

# %%
# B5.2 — Exercise
# TODO: Complete the event frame and a grounded, schema-valid record.

event_frame_B = None

student_record_B = {
    "team": None,
    "project": None,
    "event_type": None,
    "venue": None,
    "date": None,
    "project_lead": None,
}

print("Event frame:")
print(event_frame_B)

print("\nRecord:")
print(json.dumps(student_record_B, indent=2))

print("\nValidation errors:")
for error in validate_record(
    student_record_B,
    source_text_B,
    schema_B,
):
    print("•", error)

# %%
# B5 reflection
# Why does valid JSON not necessarily imply a correct extraction?
# Explain schema validity and evidence validity.

# %%
# B6 — Error analysis
# Compute P/R/F1 and identify false positives and false negatives.
# Classify boundary/type/out-of-schema errors and suggest improvements.

# %%
# B6.1 — Fixed predictions for error analysis
# TODO: Compute metrics and the false-positive/false-negative sets.

error_analysis_predictions_B = [
    ("Harbour AI", "ORG", 0, 1),                         # correct
    ("UNNC", "ORG", 6, 6),                              # correct
    ("litter-collecting boat", "PROJ", 11, 12),         # boundary error
    ("Green Campus Expo", "ORG", 14, 16),               # type error
    ("22 September", "DATE", 18, 19),                   # correct
    ("Aisha Rahman", "PER", 23, 24),                    # correct
    ("lightweight vision model", "PROJ", 30, 32),       # out-of-schema FP
]


scores_error_analysis_B = None
false_positives_B = None
false_negatives_B = None


print("Scores:", scores_error_analysis_B)
print("False positives:", false_positives_B)
print("False negatives:", false_negatives_B)

# %%
# Exit ticket
# Answer briefly: when use spans instead of BIO? What does linking add?
# Why does relation direction matter? Trigger versus argument?
# Why validate JSON? Why use exact-match scoring?

# %%
# Review
# Extract entities -> link identities -> relations/events -> records -> validate.
# Evaluate the output unit and inspect errors, not only aggregate scores.


# %%
# WRITTEN REFLECTIONS — ADD YOUR ANSWERS BELOW
# B1 — Why B-ORG/I-ORG? What happens if Expo is omitted? 
# Answer:
#
# B2 — Which entities are missed with maximum span length 2, and why?
# Answer:
#
# B3 — Identify two extraction errors and suggest improvements.
# Answer:
#
# B4 — Explain the reversed pair and missed-EcoSkimmer consequence.
# Answer:
#
# B5 — Why do JSON syntax, schema and evidence require separate checks?
# Answer:
#
# B6 — Classify each error and recommend a correction.
# Answer:
#
# Exit ticket — Respond to the six short questions above.
# Answer:
