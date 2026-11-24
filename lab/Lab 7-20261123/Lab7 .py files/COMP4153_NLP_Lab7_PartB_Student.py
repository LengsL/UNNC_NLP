# -*- coding: utf-8 -*-
# Run from your Lab7 folder: python COMP4153_NLP_Lab7_PartB_Student.py
# Dependencies: numpy, pandas, matplotlib, scikit-learn, torch.
# Install (CPU PyTorch): python -m pip install numpy pandas matplotlib scikit-learn torch
# For a CUDA-capable GPU, follow: https://pytorch.org/get-started/locally/
# Keep the required tinystories_classroom_wordlevel.txt file beside this script.
# VS Code tip: use "Run Python File" for a complete run or "Run Cell" at each # %% marker.
# Student version: Part B Tasks B6 and B7 require your own code.

# ======================================================================================
# COMP4153 Natural Language Processing
# Lab 7 — Part B — Word-level GPT-style text generation
# ======================================================================================
#
# Version: Student
# Duration: approximately 45 minutes of the original two-hour lab
# Environment: Local Python / VS Code, or Google Colab
# Mode: self-guided. Comments explain concepts and tensor shapes alongside the code.
#
# These are small educational models TRAINED FROM SCRATCH, not pretrained checkpoints.
# Part A uses bidirectional attention and one sentiment label per review.
# Part B uses causal attention and next-token prediction for word-level generation.
#
# Learning goals: split whole short stories before building the vocabulary,
# make one-token-shifted input/target windows, apply causal self-attention, train a
# word-level next-token model, inspect learning curves and generate sample stories.
# The local story corpus is original TinyStories-style classroom material, NOT an
# official TinyStories dataset extract. No data are downloaded by these scripts.

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

# Only the story corpus is required for Part B; the IMDB CSV is NOT required.
# The source notebook's B1 section below validates this file.

# ======================================================================================
# Part B — GPT-style word-level text generation with short stories
# ======================================================================================
#
# This part trains a small GPT-style model from scratch to predict the next word or
# punctuation token. The corpus contains short, straightforward English stories,
# supplied as one UTF-8 .txt file.
#
# short stories → word + punctuation tokens → token IDs
# → shifted input/target windows → token + positional embeddings
# → causal Transformer → next-token logits → cross-entropy
# → generate one token at a time
#
# In our word-level setup, a token such as garden or rabbit is predicted in one step.
# Punctuation such as . and , is also kept as a token so that the generated text can be
# displayed as readable sentences.
#
# The vocabulary is deliberately limited to frequent words from the training stories.
# This makes it possible to demonstrate word-level generation on a CPU without loading a
# pretrained GPT model.
#
# Two tasks: B6 (configure and train the provided model) and B7 (plot and interpret its
# learning curves). All dataset preparation, causal masking, the Transformer class,
# training helper and generation code are provided.
#
# --------------------------------------------------------------------------------------
# How next-token prediction differs from classification
# --------------------------------------------------------------------------------------
#
# Part A produces one decision per review. Part B instead learns from the sequence
# itself: every token position supplies a prediction target consisting of the following
# word or punctuation token. For a sequence the cat sat, the model is asked to predict
# cat from the, then sat from the cat.
#
# Self-attention remains central, but the permitted context changes. The language model
# must not access the correct next token before predicting it, so we apply a causal
# mask. This is the main architectural contrast to observe between the two parts of the
# practical.


# --------------------------------------------------------------------------------------
# B1. Load the local short-story corpus
# --------------------------------------------------------------------------------------
#
# Place tinystories_classroom_wordlevel.txt in the same Lab7 folder as this Python
# script and imdb_reviews_10000_fixed_split.csv.
#
# This is a small, original TinyStories-style classroom corpus provided with the Python
# script, rather than an official extract from the published TinyStories dataset. It
# contains 560 short stories and uses the line <eos> to mark each story's end. See
# DATASET_README.txt for provenance. The original TinyStories project is at Hugging Face
# (https://huggingface.co/datasets/roneneldan/TinyStories).
#
# Each story becomes one independent unit. Later we divide whole stories into train,
# development and test sets so no single story can appear in more than one split.
#
# --------------------------------------------------------------------------------------
# What to check after loading
# --------------------------------------------------------------------------------------
#
# The Python script reads the text using UTF-8, divides it at the literal <eos> markers,
# trims whitespace and discards empty entries. The output prints the number of stories
# and a sample from the first story. Read the sample to check that punctuation and word
# spacing look sensible before proceeding.
#
# The <eos> separator in the raw file is converted later into the special vocabulary ID
# <EOS>. That marker tells the next-token learner where a story ends; it is not a
# sentiment label and is not displayed as ordinary prose during generation.


# %%
# This file contains the local 560-story classroom corpus.
STORIES_FILE = LAB_DIR / "tinystories_classroom_wordlevel.txt"

if not STORIES_FILE.is_file():
    raise FileNotFoundError(
        f"Cannot find {STORIES_FILE.name} in {LAB_DIR}. "
        "Place the supplied text file beside the notebook."
    )

# The source has one literal <eos> separator between stories.
# Splitting on it avoids mixing the same story across data splits.
# The read_text call does not download anything: it reads local UTF-8 text.
all_story_text = STORIES_FILE.read_text(encoding="utf-8")
# Keep only nonempty stories, stripping surrounding whitespace.
stories = [part.strip() for part in all_story_text.split("<eos>") if part.strip()]

if len(stories) < 30:
    raise ValueError("The story file is unexpectedly small; check that the correct file was supplied.")

print("Story file:", STORIES_FILE.name)
print("Total stories:", len(stories))
print("File size (characters):", len(all_story_text))
print("Example story:\n", stories[0][:420])

# --------------------------------------------------------------------------------------
# B2. Make train/development/test stories and build a word-level vocabulary
# --------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------
# Step 1 — Split by story
# --------------------------------------------------------------------------------------
#
# We shuffle the story indices using a fixed seed and allocate approximately:
#
# - 80% of stories to training;
# - 10% to development (validation);
# - 10% to testing.
#
# Shuffling story indices is safe here because each story is a separate document.
# Crucially, we split before constructing training windows, avoiding text overlap across
# the three groups.
#
# --------------------------------------------------------------------------------------
# Step 2 — Tokenise words and punctuation
# --------------------------------------------------------------------------------------
#
# A lightweight regex turns:
#
# Once upon a time, Lily found a ball.
#
# into:
#
# ["once", "upon", "a", "time", ",", "lily", "found", "a", "ball", "."]
#
# The tokenizer lowercases words, keeps simple contractions as one token, and preserves
# punctuation.
#
# --------------------------------------------------------------------------------------
# Step 3 — Build the vocabulary from training stories only
# --------------------------------------------------------------------------------------
#
# Three reserved tokens have fixed IDs:
#
# <PAD>  = 0    reserved for padding
# <UNK>  = 1    word missing from the training vocabulary
# <EOS>  = 2    end of a story
#
# The other tokens are selected by frequency from the training split. GPT_MAX_VOCAB =
# 1500 sets an upper bound; the actual vocabulary can be smaller because this is a
# deliberately small corpus.
#
# Validation and test text use the same training vocabulary. Unseen tokens are converted
# to <UNK>, rather than extending the vocabulary with held-out information.
#
# --------------------------------------------------------------------------------------
# Why split before building the vocabulary?
# --------------------------------------------------------------------------------------
#
# A separate story must not be partly in training and partly in evaluation. Splitting
# the list of whole stories first prevents the same story appearing in two sets. The
# vocabulary is then constructed from training stories only; otherwise, the model could
# indirectly benefit from knowing words that occur exclusively in held-out stories.
#
# gpt_word_to_id maps tokens to integers; gpt_id_to_word reverses the mapping for
# interpreting model predictions. The special <EOS> ID is appended after every story,
# including the final story in each split. The code then concatenates the stories within
# each split into one token stream; streams from different splits are never mixed.
#
# Note: <PAD> is reserved but is not needed for the fixed-length training windows used
# below. <UNK> is still needed for words that occur in the development or test corpus
# but were absent from training.


# %%
# Split WHOLE stories before vocabulary construction or model-window creation.
story_indices = list(range(len(stories)))
random.Random(SEED).shuffle(story_indices)
# Use fixed 80/10/10 proportions on the shuffled document indices.
train_end = int(0.80 * len(story_indices))
dev_end = int(0.90 * len(story_indices))

train_stories = [stories[i] for i in story_indices[:train_end]]
dev_stories = [stories[i] for i in story_indices[train_end:dev_end]]
test_stories = [stories[i] for i in story_indices[dev_end:]]

# Words, simple contractions and punctuation are separate tokens.
# Example: "The cat's ball!" -> ["the", "cat's", "ball", "!"]
def tokenize_story(text):
    # Tokens include words and standalone punctuation, e.g. the final exclamation mark.
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", text.lower())

# These three token IDs are reserved in every split.
GPT_PAD_IDX = 0
GPT_UNK_IDX = 1
GPT_EOS_IDX = 2
GPT_MAX_VOCAB = 1500

# Build the vocabulary from the TRAINING stories only.
# Never inspect held-out stories when choosing word IDs.
word_counts = Counter(token for story in train_stories for token in tokenize_story(story))
gpt_word_to_id = {"<PAD>": GPT_PAD_IDX, "<UNK>": GPT_UNK_IDX, "<EOS>": GPT_EOS_IDX}
for word, count in word_counts.most_common(GPT_MAX_VOCAB - len(gpt_word_to_id)):
    gpt_word_to_id[word] = len(gpt_word_to_id)

# Reverse dictionary lets us display model-predicted IDs as readable tokens.
gpt_id_to_word = {index: word for word, index in gpt_word_to_id.items()}
GPT_VOCAB_SIZE = len(gpt_word_to_id)

# Append <EOS> between stories so the model observes document boundaries.
def stories_to_ids(story_list):
    ids = []
    for story in story_list:
        ids.extend(gpt_word_to_id.get(word, GPT_UNK_IDX) for word in tokenize_story(story))
        ids.append(GPT_EOS_IDX)  # Teach the model that one story has ended.
    return torch.tensor(ids, dtype=torch.long)

# Build three separate token streams; never concatenate one split with another.
word_train_ids = stories_to_ids(train_stories)
word_dev_ids = stories_to_ids(dev_stories)
word_test_ids = stories_to_ids(test_stories)

print("Train/dev/test stories:", len(train_stories), len(dev_stories), len(test_stories))
print("Vocabulary size V:", GPT_VOCAB_SIZE)
print("Train/dev/test tokens:", len(word_train_ids), len(word_dev_ids), len(word_test_ids))
print("Example tokenisation:", tokenize_story(stories[0])[:20])
print("Top vocabulary items:", word_counts.most_common(10))

# --------------------------------------------------------------------------------------
# B3. Construct next-token input/target pairs
# --------------------------------------------------------------------------------------
#
# GPT learns through self-supervised next-token prediction. If the text is:
#
# once  upon  a  time  ,  a  cat
#
# then the aligned inputs and targets are:
#
# input:   once  upon  a  time  ,  a
# target:  upon  a     time ,     a  cat
#
# For a window containing T = 32 input tokens, we need 33 consecutive tokens: 32 inputs
# and the next token for the final target.
#
# input_ids  [T]
# target_ids [T]
#
# We use a stride of T, so consecutive input windows do not overlap. The end-of-story
# marker <EOS> can appear as an input or as the next-token target.
#
# A DataLoader groups examples into:
#
# input_ids   [B,T]
# target_ids  [B,T]
#
# The model's output will have shape [B,T,V]: at each position, a score for every
# vocabulary token. Here V represents the number of word and punctuation tokens in our
# vocabulary.
#
# --------------------------------------------------------------------------------------
# Examine the one-position shift carefully
# --------------------------------------------------------------------------------------
#
# The training pair (input_ids[t], target_ids[t]) always refers to adjacent positions in
# the same token stream: the target is the token immediately following the current
# input. For example, if the input IDs correspond to once upon a, the targets are upon a
# time. Notice that target_ids is shifted one token ahead of input_ids, as the model is
# trained to predict the next token at each position.
#
# A window of length T=32 contains 32 such prediction cases. Using a non-overlapping
# stride simplifies the practical and keeps the dataset compact; incomplete tail tokens
# that cannot form a full input/target window are dropped. Some windows may cross an
# <EOS> boundary, making the story-end marker observable as part of the continuous
# training stream.
#
# At this stage input_ids and target_ids have the same shape ([B,T]), but different
# values. Later the language model turns inputs into [B,T,V] scores, and the loss
# compares these to the target IDs.


# %%
# Each window contributes T next-token prediction targets.
GPT_SEQUENCE_LENGTH = 32
GPT_BATCH_SIZE = 64

# Convert a stream into non-overlapping input windows and shifted targets.
def make_word_windows(token_stream, seq_len):
    # An input window needs ONE additional token for its final target.
    starts = range(0, len(token_stream) - seq_len, seq_len)
    # Input at index t is paired with the next token at index t+1.
    inputs = [token_stream[i:i + seq_len] for i in starts]
    targets = [token_stream[i + 1:i + seq_len + 1] for i in starts]
    # Very short datasets should fail clearly rather than producing empty loaders.
    if not inputs:
        raise ValueError("Not enough tokens to construct one window; check the dataset.")
    return TensorDataset(torch.stack(inputs), torch.stack(targets))

word_train_data = make_word_windows(word_train_ids, GPT_SEQUENCE_LENGTH)
word_dev_data = make_word_windows(word_dev_ids, GPT_SEQUENCE_LENGTH)
word_test_data = make_word_windows(word_test_ids, GPT_SEQUENCE_LENGTH)

# Randomise the order of training windows without reshuffling validation/test.
word_generator = torch.Generator().manual_seed(SEED)
word_train_loader = DataLoader(
    word_train_data,
    batch_size=GPT_BATCH_SIZE,
    shuffle=True,
    generator=word_generator,
)
word_dev_loader = DataLoader(word_dev_data, batch_size=GPT_BATCH_SIZE)
word_test_loader = DataLoader(word_test_data, batch_size=GPT_BATCH_SIZE)

# Inspect one pair to verify that the target is shifted, not copied.
example_x, example_y = word_train_data[0]
print("Number of training windows:", len(word_train_data))
print("First input/target shapes:", tuple(example_x.shape), tuple(example_y.shape))
print("Input words:", [gpt_id_to_word[i] for i in example_x[:12].tolist()])
print("Next words:", [gpt_id_to_word[i] for i in example_y[:12].tolist()])

# --------------------------------------------------------------------------------------
# B4. Why causal masking is essential
# --------------------------------------------------------------------------------------
#
# When predicting the next token after once, the model may use once but must not inspect
# upon or later tokens.
#
# For every query position t, causal attention permits keys at positions 0 through t,
# but blocks positions to its right.
#
# query / key    1   2   3   4   5
# 1              ✓   ×   ×   ×   ×
# 2              ✓   ✓   ×   ×   ×
# 3              ✓   ✓   ✓   ×   ×
# 4              ✓   ✓   ✓   ✓   ×
# 5              ✓   ✓   ✓   ✓   ✓
#
# PyTorch's Boolean attention mask uses True to mean a connection is blocked. The mask
# goes into the attention layer, which sets masked scores to negative infinity before
# softmax.
#
# Training can still predict all T next tokens in parallel because future attention is
# blocked by the mask.
#
# --------------------------------------------------------------------------------------
# How to read the mask
# --------------------------------------------------------------------------------------
#
# The mask is a T × T square. Row t is the token that asks for information (the query),
# and column j is a candidate information source (the key). In the strict upper
# triangle, j > t: those entries are True (blocked). The diagonal is unblocked, because
# a position may use its own observed token while predicting the next token.
#
# Without this mask, a token at position t could attend to position t+1, where the
# correct answer already sits in the input sequence. That would be target leakage.
# Contrast this with the padding mask in Part A, which hides meaningless [PAD] entries
# rather than future tokens.


# %%
def make_causal_mask(T, device):
    # Shape [T,T]. Diagonal and lower triangle are allowed (False).
    # Strict upper triangle: True means "cannot look into the future".
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

print("Causal mask (1 = future key blocked):")
print(make_causal_mask(5, device="cpu").int())

# --------------------------------------------------------------------------------------
# B5. Build a small GPT-style word-level model
# --------------------------------------------------------------------------------------
#
# We reuse PyTorch's nn.TransformerEncoderLayer as a causally masked self-attention
# block. Despite the PyTorch class name, this model behaves as a decoder-only GPT-style
# Transformer because every block has the causal mask and no cross-attention.
#
# input token IDs             [B,T]
#  → token embeddings         [B,T,D]
#  + position embeddings      [B,T,D]
#  → causally masked blocks   [B,T,D]
#  → vocabulary projection    [B,T,V]
#
# D is the model width (64 by default). V is the size of the word vocabulary. The
# vocabulary projection nn.Linear(D, V) produces one score for each possible next-token
# class.
#
# The model returns raw logits, not softmax probabilities. We reshape [B,T,V] into
# [B*T,V] before calculating nn.CrossEntropyLoss against flattened next-token targets of
# shape [B*T].
#
# These blocks include attention, a position-wise feed-forward network, residual
# connections, and layer normalisation. We need only one Transformer layer for the
# guided experiment.
#
# --------------------------------------------------------------------------------------
# What the output projection predicts
# --------------------------------------------------------------------------------------
#
# At each input position, a contextual vector of width D is projected to V numbers, one
# score for every possible next token in the vocabulary. If the batch contains B=64
# windows of T=32 tokens, the logits have shape [64,32,V]. Flattening the first two
# dimensions produces 64 × 32 independent next-token loss terms, while the causal
# attention block still lets earlier positions inform later ones.
#
# Implementation detail: PyTorch calls the reusable block TransformerEncoderLayer, but
# supplying an upper-triangular causal mask makes its self-attention unidirectional. We
# use it here as a compact implementation of the masked self-attention and feed-forward
# components of a GPT-style model; it is not a pretrained GPT system.


# %%
class MiniGPTWordModel(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=64, num_heads=4, ff_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.max_len = max_len  # Maximum context used for both training and generation.
        # The token and position embeddings both produce width-D vectors.
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # batch_first=True means [B,T,D] rather than [T,B,D].
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # PyTorch calls this an encoder; the CAUSAL mask below restricts it to the past.
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        if T > self.max_len:  # Never index beyond learned position embeddings.
            raise ValueError("Input exceeds the maximum Transformer context length.")

        # Word identity + token position, both in the D-dimensional space.
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Prevent each query from attending to later words.
        # Attention at position t sees at most positions 0...t.
        causal_mask = make_causal_mask(T, input_ids.device)
        x = self.blocks(x, mask=causal_mask)  # [B,T,D].
        # Each contextual vector gets one score per potential next word/token.
        return self.lm_head(x)                # [B,T,V] raw next-word logits.

# --------------------------------------------------------------------------------------
# B5.1 Guided training and evaluation functions
# --------------------------------------------------------------------------------------
#
# The same two helpers are provided as in a standard PyTorch classifier training
# workflow, with one difference: every position is a classification case and the target
# is the next word rather than a sentiment label.
#
# Each mini-batch is processed as:
#
# model(input_ids)                    → logits [B,T,V]
# reshape logits to [B*T,V]
# reshape targets to [B*T]
# CrossEntropyLoss                    → average next-token loss
# backward() + gradient clipping     → update token/attention/output weights
#
# We accumulate loss according to the number of target tokens, not the number of
# windows, then evaluate the development data once per epoch.
#
# Perplexity is:
#
# $$
# \mathrm{PPL} = \exp(\mathrm{average\ token\ cross\ entropy}).
# $$
#
# Lower perplexity is generally better on the same held-out dataset and vocabulary. It
# is not a direct score of whether a generated story is meaningful.
#
# The model completes all configured epochs and restores the best completed
# development-loss checkpoint afterward.
#
# --------------------------------------------------------------------------------------
# Cross-entropy and perplexity
# --------------------------------------------------------------------------------------
#
# Cross-entropy penalises the model when the correct next word receives a low predicted
# probability. The loss function expects raw logits; it performs the necessary
# normalisation internally. A model predicting next tokens uniformly over V
# possibilities would have cross-entropy near ln(V) and perplexity near V, but the
# precise baseline depends on the data distribution.
#
# A model may have low average perplexity while still producing repetitive text, because
# perplexity measures prediction quality on held-out next-token examples, not story
# originality or narrative coherence. Compare perplexities only when the dataset,
# vocabulary and tokenisation are the same.
#
# As in Part A, training uses model.train(), evaluation uses model.eval() and
# @torch.no_grad(), and the best development checkpoint is restored after completing all
# configured epochs.


# %%
@torch.no_grad()
def evaluate_gpt(model, loader, loss_fn):
    model.eval()  # Dropout is disabled for validation/test.
    total_loss, total_tokens = 0.0, 0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model(input_ids)  # [B,T,V]: one vocabulary score per position.
        # Cross-entropy treats every position as a separate V-class decision.
        V = logits.size(-1)
        loss = loss_fn(logits.reshape(-1, V), targets.reshape(-1))

        # Account for actual target TOKENS, not just the number of windows.
        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    # Token-weighted mean over the whole data loader.
    average_loss = total_loss / total_tokens
    return average_loss, math.exp(min(average_loss, 20))


def train_gpt(model, train_loader, dev_loader, loss_fn, optimiser, epochs=4):
    history = []
    best_dev_loss, best_state, best_epoch = float("inf"), None, None

    # Finish every epoch before restoring the best development checkpoint.
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for input_ids, targets in train_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Standard optimisation sequence, with next-word targets.
            optimiser.zero_grad()
            logits = model(input_ids)  # [B,T,V].
            V = logits.size(-1)
            loss = loss_fn(logits.reshape(-1, V), targets.reshape(-1))
            loss.backward()  # Differentiate loss through the model parameters.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            token_count = targets.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

        train_loss = total_loss / total_tokens
        # Evaluate on different stories without training on them.
        dev_loss, dev_ppl = evaluate_gpt(model, dev_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "dev_perplexity": dev_ppl})
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | dev {dev_loss:.4f} | dev PPL {dev_ppl:.2f}")

        if dev_loss < best_dev_loss:
            best_dev_loss, best_epoch = dev_loss, epoch
            # Deep-copy this epoch before later optimiser updates overwrite it.
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)  # Restore only after ALL epochs finish.
    print("Restored best completed epoch:", best_epoch)
    return pd.DataFrame(history), best_epoch

# --------------------------------------------------------------------------------------
# Task B6 — Configure and train the small word-level GPT-style model
# --------------------------------------------------------------------------------------
#
# Complete the code section by creating one model and calling the provided training
# helper. You do not need to implement the training loop.
#
# Recommended settings for this classroom corpus:
#
# GPT_D_MODEL = 64
# GPT_NUM_HEADS = 4
# GPT_FF_DIM = 128
# GPT_NUM_LAYERS = 1
# GPT_DROPOUT = 0.1
# GPT_EPOCHS = 4
# GPT_LEARNING_RATE = 0.001
#
# Steps:
#
# 1. Reset the random seed with set_seed().
# 2. Create MiniGPTWordModel with vocab_size=GPT_VOCAB_SIZE, max_len=GPT_SEQUENCE_LENGTH
#    and the settings above. Move it to device.
# 3. Create gpt_loss_fn = nn.CrossEntropyLoss() and an AdamW optimiser (learning rate
#    0.001).
# 4. Call train_gpt(...) with word_train_loader and word_dev_loader.
#
# Save the outputs as gpt_model, gpt_loss_fn, gpt_optimizer, gpt_history, and
# gpt_best_epoch. These names are needed by the later cells.
#
# The same code runs on CPU or CUDA. Run all four epochs for the default experiment;
# very short CPU trials can use fewer epochs for initial debugging only.
#
# --------------------------------------------------------------------------------------
# Hints before writing code
# --------------------------------------------------------------------------------------
#
# The class definition and functions train_gpt() and evaluate_gpt() are already
# provided. Your task is to instantiate and connect them. GPT_VOCAB_SIZE and
# GPT_SEQUENCE_LENGTH came from the earlier data-preparation cells; keep those values
# rather than typing arbitrary numbers.
#
# gpt_loss_fn must be nn.CrossEntropyLoss() because the model produces raw next-token
# logits. An AdamW optimiser needs the parameters of the same model you intend to train.
# train_gpt() returns a history table and the index of the restored checkpoint's epoch,
# so assign both return values. A correct cell enables the later test and generation
# cells to run without modification.


# %%
# Task B6: configure MiniGPTWordModel and call train_gpt().
# The model accepts input IDs [B,T] and returns next-token logits [B,T,V].
# Use the settings in the Task B6 instructions above.
# Required names: gpt_model, gpt_loss_fn, gpt_optimizer, gpt_history, gpt_best_epoch.

# YOUR CODE HERE
# Hint: write the hyperparameter variables, then instantiate gpt_model.
# The loss and optimiser must refer to the same model you are training.
# Finally call train_gpt(...) and assign both returned values.

# --------------------------------------------------------------------------------------
# Task B7 — Plot and interpret the GPT learning curves
# --------------------------------------------------------------------------------------
#
# The gpt_history DataFrame contains epoch, train_loss, dev_loss, and dev_perplexity.
#
# 1. Figure 1: plot training and development cross-entropy losses against epoch on the
#    same axes, with a legend.
# 2. Figure 2: plot development perplexity against epoch on a separate figure.
# 3. Answer: Does development loss improve? Do the curves diverge? Which completed epoch
#    was restored?
#
# Since perplexity is the exponential of cross-entropy, these two development curves
# move in the same direction. They have different numeric scales.
#
# --------------------------------------------------------------------------------------
# What the two figures tell you
# --------------------------------------------------------------------------------------
#
# In Figure 1, plot the training and development losses on the same scale to inspect
# both learning and generalisation. In Figure 2, plot development perplexity to express
# next-token prediction performance in a more interpretable, exponential scale. Both
# figures should use the epochs already recorded in gpt_history.
#
# The selected checkpoint is the epoch with the smallest development loss, not
# necessarily the lowest training loss or the last epoch. Because perplexity is
# exp(dev_loss), it selects the same best epoch. State whether the development curve
# improved across training and whether signs of overfitting appeared.


# %%
# Task B7: plot two figures from gpt_history.
# Figure 1: train_loss and dev_loss against epoch (with legend).
# Figure 2: dev_perplexity against epoch.
# Add titles and axis labels. Do not retrain the model.

# YOUR CODE HERE
# Hint: gpt_history is already a DataFrame; plot its columns directly.
# Figure 1 needs TWO labelled curves; Figure 2 needs ONE curve.
# Remember plt.show() after each figure.

# --------------------------------------------------------------------------------------
# B8. Evaluate on held-out stories
# --------------------------------------------------------------------------------------
#
# train_gpt() has restored the checkpoint that achieved the lowest completed-epoch
# development loss. Now evaluate the unseen test stories once.
#
# We report average next-word cross-entropy and perplexity. These measure next-token
# prediction on unseen text, not guaranteed story quality.
#
# The test set was not used in train_gpt() to update parameters or select a checkpoint.
# Thus, the following cell provides a final evaluation of the development-selected model
# on different, unseen stories.
#
# Lower test loss and perplexity generally indicate stronger next-token prediction for
# this corpus and its vocabulary.


# %%
# B8 depends on Task B6; preserve the TODO while allowing a first full-file run.
# Once training is configured and executed, this section evaluates the test set.
_required_B6 = ("gpt_model", "gpt_loss_fn", "gpt_history", "gpt_best_epoch")
if all(name in globals() for name in _required_B6):
    # Evaluate only after train_gpt() has restored the best development checkpoint.
    gpt_test_loss, gpt_test_ppl = evaluate_gpt(
        gpt_model,
        word_test_loader,
        gpt_loss_fn,
    )
    print(f"Short-story test next-word loss: {gpt_test_loss:.4f}")
    print(f"Short-story test perplexity: {gpt_test_ppl:.2f}")
else:
    print("Complete Task B6 before running the held-out test evaluation (B8).")

# --------------------------------------------------------------------------------------
# B9. Generate a story one word/token at a time
# --------------------------------------------------------------------------------------
#
# During training, all positions are processed in parallel under the causal mask. During
# generation, future text is unknown, so each generated token is appended and used in
# the next step.
#
# prompt tokens → last-position vocabulary logits
# → divide logits by temperature → softmax → sample one token
# → append the sampled token → repeat
#
# The model sees at most GPT_SEQUENCE_LENGTH recent tokens. Each output step predicts an
# entire word (or punctuation token).
#
# Temperature: values below 1 favour high-probability words; values above 1 make
# sampling more varied. Generation may stop when <EOS> is sampled, representing an
# end-of-story boundary.
#
# The detokenize_story() function joins words and removes spaces before common
# punctuation, turning tokens such as ['a','cat','.', 'she','ran','!'] into a readable
# sentence.
#
# Since this is a small model trained from scratch on a limited, formulaic corpus, it
# may repeat phrases or create awkward sentences. This is expected.
#
# --------------------------------------------------------------------------------------
# Follow one generation step
# --------------------------------------------------------------------------------------
#
# The prompt is tokenised with the same function and vocabulary used for training. The
# most recent GPT_SEQUENCE_LENGTH token IDs are passed to the model. From its [1,T,V]
# output, we take only [1,-1,:]: the scores at the last position, which predict the next
# token. After applying temperature and softmax, sampling picks one vocabulary ID, which
# becomes part of the next context.
#
# <PAD> and <UNK> are excluded from sampled outputs, and generation stops early when
# <EOS> appears. When the context grows beyond the training length, it is truncated to
# the most recent 32 token IDs. The position indices therefore start again at zero for
# that truncated context, matching this simplified model's fixed positional embedding
# scheme.
#
# Interpretation task: compare the samples at temperatures 0.7 and 1.0. Lower
# temperature often gives more predictable wording, while higher temperature increases
# variety. There is no guarantee that either sample will be grammatical or coherent
# after this brief classroom training run.


# %%
def detokenize_story(words):
    # Simple readable formatting: remove spaces before punctuation.
    # Treat punctuation separately during learning, but join it naturally for display.
    result = " ".join(words)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    result = re.sub(r"\s+'", "'", result)
    return result


@torch.no_grad()
def generate_words(model, prompt="once upon a time", new_tokens=70, temperature=0.8):
    # Avoid degenerate sampling requests before creating model inputs.
    if not prompt.strip():
        raise ValueError("Prompt must contain at least one word.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Use the exact same tokenizer/vocabulary as training.
    # Use the training tokenizer, including lowercase and punctuation handling.
    prompt_words = tokenize_story(prompt)
    generated_ids = [gpt_word_to_id.get(word, GPT_UNK_IDX) for word in prompt_words]
    model.eval()

    for _ in range(new_tokens):
        # Use the most recent tokens only: learned positions range 0..max_len-1.
        visible_ids = generated_ids[-model.max_len:]
        context = torch.tensor([visible_ids], dtype=torch.long, device=device)

        # Only the final visible position predicts the next token.
        # Final position predicts the next token; temperature controls sharpness.
        next_logits = model(context)[:, -1, :] / temperature
        # Never display padding or unknown-token placeholders as generated prose.
        next_logits[:, GPT_PAD_IDX] = float("-inf")
        next_logits[:, GPT_UNK_IDX] = float("-inf")
        probabilities = torch.softmax(next_logits, dim=-1)
        # Sample an ID rather than always taking argmax.
        next_id = torch.multinomial(probabilities, num_samples=1).item()

        # Stop at a learned story boundary.
        if next_id == GPT_EOS_IDX:
            break  # End of the generated story.
        generated_ids.append(next_id)

    # Avoid printing special-token placeholders as ordinary words.
    words = [gpt_id_to_word[index] for index in generated_ids if index not in (GPT_PAD_IDX, GPT_UNK_IDX, GPT_EOS_IDX)]
    return detokenize_story(words)



# Generate only after B6 has created and trained the student model.
if all(name in globals() for name in ("gpt_model", "gpt_loss_fn", "gpt_history", "gpt_best_epoch")):
    set_seed()
    for temperature in [0.7, 1.0]:
        print("\n" + "=" * 52)
        print("Temperature:", temperature)
        print(generate_words(gpt_model, prompt="once upon a time", new_tokens=70, temperature=temperature))

else:
    print("Complete Task B6 before running sample generation (B9).")
    print("Complete Task B7 to create the two learning-curve figures.")

# --------------------------------------------------------------------------------------
# B10. Compare bidirectional classification and causal generation
# --------------------------------------------------------------------------------------
#
# | Property | Part A — IMDB | Part B — Short stories |
# |---|---|---|
# | Input units | Words with [CLS] and [SEP] | Words and punctuation, with <EOS> |
# | Attention | Bidirectional encoder | Causally masked self-attention |
# | Training target | One sentiment class per review | Next token at every position |
# | Model output | [B,2] sentiment logits | [B,T,V] token logits |
# | Prediction | argmax over two classes | Sampling next token and appending it |
# | Evaluation | Accuracy, macro F1, confusion matrix | Next-token loss and perplexity |
#
# Both models use token embeddings, position embeddings, multi-head self-attention,
# feed-forward layers, residual connections and layer normalisation. The crucial
# difference is which positions can attend to which other positions, and what the model
# is trained to predict.
#
# --------------------------------------------------------------------------------------
# Optional experiments
# --------------------------------------------------------------------------------------
#
# - Try GPT_D_MODEL = 96 and compare development loss against the default of 64. Be sure
#    GPT_D_MODEL is divisible by GPT_NUM_HEADS.
# - Change GPT_EPOCHS from 4 to 6, then compare development loss and generated samples.
# - Generate from "one day" or "the little dog", or compare temperatures 0.7 and 1.0.
#
# Do not change the train/development/test splits when comparing models. Use development
# loss for architecture choices.
#
# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------
#
# - Vaswani et al. (2017), Attention Is All You Need.
# - Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers for
#    Language Understanding.
# - Radford et al. (2018), Improving Language Understanding by Generative Pre-Training.
# - Eldan and Li (2023), TinyStories: How Small Can Language Models Be and Still Speak
#    Coherent English?, original dataset
#    (https://huggingface.co/datasets/roneneldan/TinyStories). The supplied classroom
#    story corpus is original TinyStories-style text, not an official TinyStories
#    extract.


# End of Lab 7 Part B (Student).
