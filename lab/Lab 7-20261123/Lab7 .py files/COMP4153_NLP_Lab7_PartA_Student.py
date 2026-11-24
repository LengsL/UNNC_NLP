# -*- coding: utf-8 -*-
# Run from your Lab7 folder: python COMP4153_NLP_Lab7_PartA_Student.py
# Dependencies: numpy, pandas, matplotlib, scikit-learn, torch.
# Install (CPU PyTorch): python -m pip install numpy pandas matplotlib scikit-learn torch
# For a CUDA-capable GPU, follow: https://pytorch.org/get-started/locally/
# Keep the required imdb_reviews_10000_fixed_split.csv file beside this script.
# VS Code tip: use "Run Python File" for a complete run or "Run Cell" at each # %% marker.

# ======================================================================================
# COMP4153 Natural Language Processing
# Lab 7 — Part A — BERT-style IMDB sentiment classification
# ======================================================================================
#
# Version: Student
# Duration: approximately 70 minutes of the original two-hour lab
# Environment: Local Python / VS Code, or Google Colab
# Mode: self-guided. Comments explain concepts and tensor shapes alongside the code.
#
# These are small educational models TRAINED FROM SCRATCH, not pretrained checkpoints.
# Part A uses bidirectional attention and one sentiment label per review.
# Part B uses causal attention and next-token prediction for word-level generation.
#
# Learning goals: convert IMDB reviews to token IDs, build a training-only
# vocabulary, encode padding masks, train a small bidirectional Transformer with [CLS],
# restore the best development-loss epoch, and evaluate on the untouched test set.

# %%
# ======================================================================================
# 0. Setup — imports, reproducibility and local data files
# ======================================================================================
# Python's standard modules are used for paths, token counts, random seeds and regex.
from pathlib import Path
from collections import Counter
import copy
import math
import os
import random
import re

# NumPy and pandas handle arrays/tables. Matplotlib shows learning curves.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch supplies embeddings, attention layers, datasets and optimisation.
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Part A uses the classification metrics; importing them for B is harmless.
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

SEED = 42

def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch for comparable classroom experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
# All input tensors and model weights must be on the same CPU or CUDA device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Device:", device)

# Colab uses Google Drive; local VS Code uses the SCRIPT'S folder, not the terminal's
# possibly different working directory. Keep dataset files beside the .py file.
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    LAB_DIR = Path("/content/drive/MyDrive/COMP4153/Lab7")
else:
    LAB_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

if not LAB_DIR.is_dir():
    raise FileNotFoundError(f"Lab folder not found: {LAB_DIR}")
# Relative paths in later sections now point at the same folder.
os.chdir(LAB_DIR)
print("Lab folder:", LAB_DIR)

# Dataset requirement: fixed IMDB train/dev/test split. Never recreate this split.
IMDB_FILE = LAB_DIR / "imdb_reviews_10000_fixed_split.csv"
if not IMDB_FILE.is_file():
    raise FileNotFoundError(
        f"IMDB file missing: {IMDB_FILE}. Place the CSV beside this .py script."
    )
print("IMDB CSV found:", IMDB_FILE.name)

# CLI-friendly substitute for Jupyter's display(): the table prints to the terminal.
def display(value):
    if hasattr(value, "to_string"):
        print(value.to_string())
    else:
        print(value)

# ======================================================================================
# Part A — Guided: BERT-style IMDB sentiment classification
# ======================================================================================
#
# A BERT-style encoder reads the entire review before predicting sentiment:
#
# review text → word IDs with [CLS] and [SEP] → embedding + position vectors
# → bidirectional self-attention + FFN → [CLS] contextual vector
# → two logits (negative / positive)
#
# Unlike GPT-style generation, the encoder can use words appearing before and after any
# position. The classification decision uses the final state of the special first token
# [CLS].
#
# --------------------------------------------------------------------------------------
# Why does this model use an encoder?
# --------------------------------------------------------------------------------------
#
# Sentiment is a sequence-level prediction: we want one label for the entire review, not
# a new token after each word. Bidirectional self-attention allows each review token to
# incorporate information from the surrounding review. This matters for expressions such
# as “not very good”, where an isolated positive word does not determine the overall
# sentiment.
#
# Inside self-attention, each token representation produces a query, key and value.
# Similarity between queries and keys determines how strongly information from the
# values is combined. PyTorch implements multi-head self-attention in the Transformer
# layer below.
#
# [CLS] is an artificial token placed at the beginning of every review. After the
# encoder processes the sequence, its output vector at position 0 is fed into a
# classifier with two outputs. BERT-style here means an encoder with bidirectional
# self-attention; it does not mean that the original pretrained BERT weights or
# WordPiece tokenizer are used.


# --------------------------------------------------------------------------------------
# A1. Load the fixed IMDB splits
# --------------------------------------------------------------------------------------
#
# The CSV contains 10,000 reviews and the predefined train, dev, test labels. Do not
# reshuffle the splits:
#
# - Train: fit weights and build the word vocabulary.
# - Development: evaluate after each epoch and select the lowest-development-loss
#    checkpoint.
# - Test: report final unseen-example metrics after selecting the checkpoint.
#
# A review is positive or negative. We map negative → 0 and positive → 1.


# %%
# Read the supplied reviews; preserve their existing split assignments.
imdb_df = pd.read_csv(IMDB_FILE)
# All three columns are needed later for tokenisation, targets and evaluation.
required_columns = {"text", "label", "split"}
if not required_columns.issubset(imdb_df.columns):
    raise ValueError(f"CSV must have columns: {sorted(required_columns)}")

# Boolean filtering does not reshuffle or relabel examples.
imdb_train_df = imdb_df[imdb_df["split"] == "train"].reset_index(drop=True)
imdb_dev_df = imdb_df[imdb_df["split"] == "dev"].reset_index(drop=True)
imdb_test_df = imdb_df[imdb_df["split"] == "test"].reset_index(drop=True)

print("Train / dev / test:", len(imdb_train_df), len(imdb_dev_df), len(imdb_test_df))
# Cross-tabulate the split sizes by sentiment to inspect balance.
display(pd.crosstab(imdb_df["split"], imdb_df["label"]))

# --------------------------------------------------------------------------------------
# A2. Tokenise words and build a vocabulary
# --------------------------------------------------------------------------------------
#
# The word-level tokenizer lowercases a review and keeps words/numbers. We deliberately
# use a small word vocabulary, not BERT's original WordPiece tokenizer.
#
# Special tokens:
#
# | Token | Role |
# |---|---|
# | [PAD] | Fills shorter reviews; must be ignored as an attention key |
# | [UNK] | Replaces unknown words |
# | [CLS] | First position; its final state represents the whole review |
# | [SEP] | End marker for retained review words |
#
# [MASK] is used in BERT's original masked-language-modelling pretraining, but we do not
# run MLM here. Our model is fitted directly to the IMDB sentiment labels.
#
# Building the vocabulary only on the training split prevents development/test words
# from changing its entries.
#
# --------------------------------------------------------------------------------------
# From text to integers
# --------------------------------------------------------------------------------------
#
# Neural networks operate on numbers rather than raw strings. First, the tokenizer
# extracts lowercase word tokens; then the Counter ranks words in the training data by
# frequency. Frequently occurring words receive their own integer IDs, while an unseen
# word is assigned [UNK]. Every special token also has a fixed ID, ensuring consistent
# behaviour across the three splits.
#
# For a review longer than the model's limit, we retain the beginning of the review
# after reserving room for [CLS] and [SEP]. This is a simple teaching choice: truncation
# reduces computation but can discard sentiment information appearing later in a long
# review. The embedding layer will turn these integers into learned D-dimensional
# vectors.


# %%
# Reserve indices 0–3 BEFORE assigning IDs to ordinary training words.
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
PAD_IDX, UNK_IDX, CLS_IDX, SEP_IDX = range(4)
IMDB_VOCAB_SIZE = 10000
IMDB_MAX_LEN = 96   # Includes [CLS] and [SEP]. Padding/truncation uses this length.

# Basic tokenisation; no extra NLTK downloads are required.
def tokenize_imdb(text):
    # Remove HTML-like tags that appear in some IMDB reviews.
    clean = re.sub(r"<[^>]+>", " ", str(text)).lower()
    # Extract simple words, contractions and numbers; punctuation is omitted here.
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", clean)

# Build token counts from TRAINING reviews only.
# Using dev/test text here would leak information about unseen vocabulary.
counts = Counter(token for text in imdb_train_df["text"] for token in tokenize_imdb(text))
imdb_vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
for token, frequency in counts.most_common():
    if len(imdb_vocab) >= IMDB_VOCAB_SIZE:
        break
    # Avoid allocating embeddings to singleton words in this small lab corpus.
    if frequency >= 2 and token not in imdb_vocab:
        imdb_vocab[token] = len(imdb_vocab)

print("Vocabulary size:", len(imdb_vocab))
print("Most frequent words:", counts.most_common(10))

# --------------------------------------------------------------------------------------
# A2.1 Encode reviews and make mini-batches
# --------------------------------------------------------------------------------------
#
# For a review, we form:
#
# [CLS] + up to 94 word IDs + [SEP] + [PAD] if needed
#
# Each review produces three items:
#
# input_ids       [T]   vocabulary indices
# attention_mask  [T]   1=real/special position, 0=padding
# label           []    0=negative, 1=positive
#
# TensorDataset stores these tensors without requiring a custom Dataset class. After
# batching, the shapes are [B,T], [B,T], and [B].
#
# --------------------------------------------------------------------------------------
# Padding masks and sequence lengths
# --------------------------------------------------------------------------------------
#
# All reviews in a batch must have equal tensor length. An encoded review therefore has
# exactly IMDB_MAX_LEN = 96 positions, including its special tokens and any [PAD]
# entries. The attention mask has 1 for meaningful positions and 0 for padding. In the
# model, we invert it to build the Boolean src_key_padding_mask required by PyTorch.
#
# For example, a short review might begin with [CLS] good film [SEP] [PAD] ...; its
# attention mask begins with four ones and then zeros. Padding masks are not causal
# masks: Part A still allows a real token to attend to any other real token, including
# later words.
#
# After DataLoader groups 64 examples, the tensors normally have shapes input_ids
# [64,96], attention_mask [64,96], and labels [64] (the last mini-batch may be smaller).


# %%
# The class-to-index mapping must stay identical for all three splits.
LABEL_TO_ID = {"negative": 0, "positive": 1}

# Keep only max_len - 2 words, reserving [CLS] and [SEP] positions.
def encode_review(text, vocab, max_len):
    # Out-of-vocabulary words are represented by the reserved [UNK] index.
    word_ids = [vocab.get(word, UNK_IDX) for word in tokenize_imdb(text)]
    # Reserve the first and last meaningful positions for special tokens.
    ids = [CLS_IDX] + word_ids[:max_len - 2] + [SEP_IDX]
    # A 1 tells the encoder that this is a meaningful input position.
    mask = [1] * len(ids)
    # Pad EVERY review to max_len so a batch can form rectangular tensors.
    pad_count = max_len - len(ids)
    return ids + [PAD_IDX] * pad_count, mask + [0] * pad_count

# Each example stores (IDs, attention mask, sentiment label).
def make_imdb_dataset(frame, vocab, max_len):
    examples = [encode_review(text, vocab, max_len) for text in frame["text"]]
    ids = torch.tensor([item[0] for item in examples], dtype=torch.long)
    masks = torch.tensor([item[1] for item in examples], dtype=torch.long)
    labels = torch.tensor([LABEL_TO_ID[str(x).lower()] for x in frame["label"]], dtype=torch.long)
    # Shapes are [N,T], [N,T] and [N] before the DataLoader batches them.
    return TensorDataset(ids, masks, labels)

imdb_train_data = make_imdb_dataset(imdb_train_df, imdb_vocab, IMDB_MAX_LEN)
imdb_dev_data = make_imdb_dataset(imdb_dev_df, imdb_vocab, IMDB_MAX_LEN)
imdb_test_data = make_imdb_dataset(imdb_test_df, imdb_vocab, IMDB_MAX_LEN)

IMDB_BATCH_SIZE = 64
# Shuffle training examples, not development/test examples.
train_generator = torch.Generator().manual_seed(SEED)
imdb_train_loader = DataLoader(imdb_train_data, batch_size=IMDB_BATCH_SIZE, shuffle=True, generator=train_generator)
imdb_dev_loader = DataLoader(imdb_dev_data, batch_size=128)
imdb_test_loader = DataLoader(imdb_test_data, batch_size=128)

sample_ids, sample_masks, sample_labels = next(iter(imdb_train_loader))
print("Batch token IDs:", tuple(sample_ids.shape))
print("Batch masks:", tuple(sample_masks.shape))
print("Batch labels:", tuple(sample_labels.shape))
print("First review has", int(sample_masks[0].sum()), "non-padding positions")

# --------------------------------------------------------------------------------------
# A3. Build a small BERT-style encoder
# --------------------------------------------------------------------------------------
#
# Instead of implementing every attention projection manually, we use PyTorch's
# nn.TransformerEncoderLayer. It already contains:
#
# multi-head self-attention → residual connection + LayerNorm
# → feed-forward network (FFN) → residual connection + LayerNorm
#
# It accepts an input shaped [B,T,D] and returns [B,T,D].
#
# BERT-style classifier data flow:
#
# | Stage | Tensor shape |
# |---|---|
# | Word IDs | [B,T] |
# | Word embeddings plus position embeddings | [B,T,D] |
# | Bidirectional Transformer layer | [B,T,D] |
# | Take final contextual state at position 0 ([CLS]) | [B,D] |
# | Final classifier | [B,2] |
#
# Why add positional embeddings? Attention can mix tokens irrespective of ordering;
# learned position embeddings provide information about where each token occurs. Why
# pass src_key_padding_mask? Padding tokens should not act as keys that other tokens
# attend to.
#
# Our default: D=48, four heads (12 features per head), one encoder layer, and FF=96.
# These modest values make a CPU run more practical. This is not pretrained BERT: it is
# a small encoder trained from scratch on sentiment labels.
#
# --------------------------------------------------------------------------------------
# Follow a review through the encoder
# --------------------------------------------------------------------------------------
#
# 1. Embedding lookup: each integer word ID becomes a learned vector of width D. The
#    position embedding supplies a different vector for each position, and these vectors
#    are added.
# 2. Multi-head attention: each attention head learns a different way to combine
#    information across valid review positions. With D=48 and four heads, each head has
#    12 features.
# 3. Feed-forward network: a small nonlinear network transforms each contextualised
#    position. Residual connections and layer normalisation help training remain stable.
# 4. Sequence representation: contextual[:, 0, :] extracts the output at [CLS], producing
#    [B,D].
# 5. Classification: a linear layer maps the [CLS] representation to [B,2] logits; higher
#    logits indicate the model's preferred class, but logits themselves are not
#    probabilities.
#
# nn.TransformerEncoderLayer hides the internal projection matrices, letting us focus on
# the role of the encoder. Notice that the model does not receive a causal attention
# mask. Its only mask excludes padding keys.


# %%
class MiniBERTClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=48, num_heads=4, ff_dim=96, num_layers=1, dropout=0.1):
        super().__init__()
        # Word identity and absolute position have the same dimension D.
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # PyTorch supplies multi-head self-attention, FFN, residuals and LayerNorm.
        # One layer combines heads of attention with an FFN, residuals and normalisation.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, 2)  # Two raw sentiment logits.

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape  # B reviews, each padded/truncated to T tokens.
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1,T].
        # Broadcasting adds [1,T,D] position vectors to [B,T,D] word vectors.
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # PyTorch uses True for padding positions that must not be attended to.
        pad_mask = attention_mask == 0
        # Bidirectional: any real token may attend to earlier AND later real tokens.
        contextual = self.encoder(x, src_key_padding_mask=pad_mask)  # [B,T,D].

        # [CLS] has accumulated context from the full unpadded review.
        cls_vector = contextual[:, 0, :]  # [B,D] from position zero [CLS].
        return self.classifier(cls_vector)  # [B,2], logits (no softmax).

# --------------------------------------------------------------------------------------
# A4. Training and evaluation (guided)
# --------------------------------------------------------------------------------------
#
# For each mini-batch:
#
# zero_grad() → forward pass → CrossEntropyLoss → backward()
# → gradient clipping → optimiser.step()
#
# CrossEntropyLoss accepts raw [B,2] logits with [B] target class IDs: do not apply
# softmax first.
#
# The evaluation function calls model.eval() and uses @torch.no_grad() so weights are
# not updated.
#
# After every completed epoch, we calculate development loss. We always finish the
# configured number of epochs, then restore the model state with the lowest completed
# development loss.
#
# --------------------------------------------------------------------------------------
# What changes between training and evaluation?
# --------------------------------------------------------------------------------------
#
# During training, model.train() enables dropout, while loss.backward() computes
# gradients and optimiser.step() updates weights. In evaluation, model.eval() disables
# dropout and torch.no_grad() avoids building a gradient graph. Merely calling eval()
# does not calculate a metric: the helper still needs to loop over batches and compare
# predicted labels with the true labels.
#
# The loss for each batch is a mean. To report an overall review-level mean when the
# final batch is smaller, the helper multiplies each batch loss by its number of
# reviews, adds these contributions, and divides by the total number of reviews.
#
# Gradient clipping caps the global gradient norm at 1.0 to avoid excessively large
# parameter updates. A deep copy of the best model state is kept in memory; otherwise,
# continued training would overwrite the parameters from the best development epoch.


# %%
@torch.no_grad()
def evaluate_classifier(model, loader, loss_fn):
    model.eval()  # Use evaluation behaviour; no stochastic dropout.
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_predictions, all_labels = [], []

    for input_ids, mask, labels in loader:
        input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
        logits = model(input_ids, mask)        # [B,2].
        loss = loss_fn(logits, labels)         # Mean batch loss.
        predicted = logits.argmax(dim=1)      # [B].

        # Weight mean batch loss by the number of reviews in this batch.
        B = labels.size(0)
        total_loss += loss.item() * B         # Weight by batch size.
        total_correct += (predicted == labels).sum().item()
        total_count += B
        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total_count, total_correct / total_count, all_predictions, all_labels


def train_classifier(model, train_loader, dev_loader, loss_fn, optimiser, epochs=4):
    history = []
    best_loss, best_epoch, best_state = float("inf"), None, None

    # Complete all requested epochs; this is NOT an early-stopping loop.
    for epoch in range(1, epochs + 1):
        model.train()  # Turn on dropout during parameter updates.
        train_loss_sum, train_count = 0.0, 0

        for input_ids, mask, labels in train_loader:
            input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
            optimiser.zero_grad()             # Clear earlier gradients.
            logits = model(input_ids, mask)   # Forward pass [B,2].
            loss = loss_fn(logits, labels)    # Classification error.
            loss.backward()                  # Backpropagation.
            # Clip oversized gradients before the optimiser modifies weights.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()                 # Update trainable weights.

            B = labels.size(0)
            train_loss_sum += loss.item() * B
            train_count += B

        train_loss = train_loss_sum / train_count
        # Development evaluation never updates model weights.
        dev_loss, dev_accuracy, _, _ = evaluate_classifier(model, dev_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_accuracy": dev_accuracy})
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | dev {dev_loss:.4f} | dev accuracy {dev_accuracy:.3f}")

        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            # Store a snapshot, not a reference that later training can modify.
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)  # Restore after ALL epochs are completed.
    print("Restored best completed epoch:", best_epoch)
    return pd.DataFrame(history), best_epoch

# --------------------------------------------------------------------------------------
# A5. Train the IMDB classifier
# --------------------------------------------------------------------------------------
#
# For easier CPU execution, the small encoder starts with one Transformer layer and four
# epochs. You may compare larger architectures later, but the aim here is to understand
# the workflow.
#
# IMDB_BATCH_SIZE=64, D_MODEL=48, NUM_HEADS=4, FF_DIM=96 and IMDB_EPOCHS=4. D_MODEL must
# be divisible by NUM_HEADS.
#
# Changing these settings requires creating a fresh model and optimiser, then rerunning
# the training cell.
#
# --------------------------------------------------------------------------------------
# Reading the training output
# --------------------------------------------------------------------------------------
#
# Each epoch reports a training loss, a development loss and a development accuracy.
# Training loss is computed on examples the optimiser has already seen; development
# metrics are calculated with parameter updates disabled. The checkpoint with the lowest
# development loss is restored only after every planned epoch finishes.


# %%
# All four attention heads divide the D_MODEL features equally.
D_MODEL = 48
NUM_HEADS = 4
FF_DIM = 96
NUM_LAYERS = 1
IMDB_EPOCHS = 4

# Reset initialisation before training so reruns are easier to compare.
set_seed()
imdb_model = MiniBERTClassifier(
    vocab_size=len(imdb_vocab),
    max_len=IMDB_MAX_LEN,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=NUM_LAYERS,
    dropout=0.1,
).to(device)
imdb_loss_fn = nn.CrossEntropyLoss()
imdb_optimizer = torch.optim.AdamW(imdb_model.parameters(), lr=0.0005, weight_decay=0.0001)
# The helper RETURNS the completed-epoch history and chosen development checkpoint.
imdb_history, imdb_best_epoch = train_classifier(
    imdb_model,
    imdb_train_loader,
    imdb_dev_loader,
    imdb_loss_fn,
    imdb_optimizer,
    epochs=IMDB_EPOCHS,
)

# --------------------------------------------------------------------------------------
# A6. Plot learning curves
# --------------------------------------------------------------------------------------
#
# Compare training loss (how well the model fits the seen reviews) with development loss
# (unseen review generalisation). Development accuracy gives an additional task-oriented
# measure. A decreasing training loss with increasing development loss can indicate
# overfitting. The plots use the completed-epoch values stored in imdb_history.


# %%
# Plot recorded metrics ONLY; no additional parameter updates occur here.
fig, ax = plt.subplots(figsize=(7, 4))
# A gap between train and dev loss may indicate overfitting.
ax.plot(imdb_history["epoch"], imdb_history["train_loss"], marker="o", label="train loss")
ax.plot(imdb_history["epoch"], imdb_history["dev_loss"], marker="o", label="dev loss")
ax.set(xlabel="Epoch", ylabel="Cross-entropy", title="IMDB: learning curves")
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(imdb_history["epoch"], imdb_history["dev_accuracy"], marker="o")
ax.set(xlabel="Epoch", ylabel="Development accuracy", title="IMDB: development accuracy")
ax.set_ylim(0, 1)
plt.show()

# --------------------------------------------------------------------------------------
# A7. Evaluate on the untouched IMDB test split
# --------------------------------------------------------------------------------------
#
# After selecting the best development checkpoint, evaluate once on the independent test
# split:
#
# - Accuracy: fraction of reviews predicted correctly.
# - Macro F1: average of the F1 scores for negative and positive classes.
# - Confusion matrix: number of correct/incorrect predictions for each class.
#
# The final logits [B,2] are converted to class IDs with argmax(dim=1).
#
# --------------------------------------------------------------------------------------
# Interpreting the metrics
# --------------------------------------------------------------------------------------
#
# In a binary confusion matrix, rows represent true labels and columns represent
# predicted labels. A diagonal cell is a correct prediction, while an off-diagonal cell
# shows a mistake. Macro F1 calculates an F1 score for each class separately and gives
# the two classes equal weight, which is useful when the class frequencies differ.


# %%
# Report metrics once, using the development-selected checkpoint.
imdb_test_loss, imdb_test_accuracy, imdb_predictions, imdb_labels = evaluate_classifier(
    imdb_model,
    imdb_test_loader,
    imdb_loss_fn,
)
imdb_f1 = f1_score(imdb_labels, imdb_predictions, average="macro")
print(f"Test loss:     {imdb_test_loss:.4f}")
print(f"Test accuracy: {imdb_test_accuracy:.3f}")
print(f"Macro F1:      {imdb_f1:.3f}")

cm = confusion_matrix(imdb_labels, imdb_predictions, labels=[0, 1])
# Display true labels along rows and predictions along columns.
ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot()
plt.title("IMDB: BERT-style classifier")
plt.show()

# End of Lab 7 Part A (Student).
