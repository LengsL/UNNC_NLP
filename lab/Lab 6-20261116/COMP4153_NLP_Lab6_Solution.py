# -*- coding: utf-8 -*-
# COMP4153 NLP — Lab 6: museum question answering
# Run the file as Python, or execute individual # %% cells in an editor.

# %% [markdown]
# # COMP4153 Natural Language Processing
# ## Lab 6 — Build a museum question-answering system
#
# **The task.** Visitors to the fictional Northbridge Science Museum ask about admission, opening times, exhibitions and facilities. You will build a small question-answering application that searches the museum guide, finds relevant passages and returns an answer with its source.
#
# Run the notebook from top to bottom. The first part prepares the guide, the middle part builds the search and answering functions, and the final part lets you compare different settings.
#
# ### By the end, you should be able to
#
# - Find relevant passages using **TF-IDF** and **BM25**.
# - Understand how passage IDs, metadata and indexed words are used.
# - Trace a question through **retrieval → answer extraction → source attribution**.
# - Compare retrieval with **Hit@k, Recall@k, MRR and nDCG@k**, and compare answers with **Exact Match and Token F1**.
#
# The answer-extraction component uses simple patterns for the museum's short factual answers. It is a different approach from the trained start/end predictor discussed in the lecture.

# %% [markdown]
# ### How the application works
#
# ```text
# PREPARATION (once)                          ANSWERING (each question)
# Museum notices + metadata                   Visitor's question
#           |                                         |
#           v                                         v
# Give each passage a source ID               Choose TF-IDF or BM25
#           |                                         |
#           v                                         v
# Tokenise the passage text     -------------> Rank matching passages
# and build the search index                         |
#                                                    v
#                                        Keep the highest-ranked k
#                                                    |
#                                                    v
#                                        Try to extract a short answer
#                                                    |
#                                                    v
#                                        Print answer and source ID
# ```
#
# **Two parts have different jobs.** The **retriever** ranks passages to decide *where to look*. The **extractor** examines the chosen passages to decide *what short answer, if any, can be returned*. Later, you will evaluate each part separately.

# %%
# These modules provide basic mathematics, text matching, and word counts.
import math
import re
from collections import Counter

print('Museum QA lab ready.')

# %% [markdown]
# ## Part 1 — Prepare the collection
#
# The guide has **18 short passages**. Each has an ID, a section (metadata), a title and the original text. In a real retrieval system, long documents would often be split into chunks first. Here the notices are already short enough to act as our searchable passages.
#
# The source information is important: an answer is more useful when the visitor can identify the passage supporting it.

# %%
# The museum guide is a list of short passages. Each tuple stores:
# (passage ID, museum section, notice title, full notice text).
ROWS = [('P01', 'Admission', 'Weekend adult tickets', 'Weekend admission at the Northbridge Science Museum costs £14 for adults.'),
 ('P02', 'Admission', 'Weekday adult tickets', 'Weekday admission at the Northbridge Science Museum costs £11 for adults.'),
 ('P03', 'Admission', 'Student discount', 'Students with a valid ID receive a £3 discount on museum entry.'),
 ('P04', 'Planetarium', 'Saturday show', 'The planetarium show begins at 14:30 on Saturdays.'),
 ('P05', 'Planetarium', 'Weekday show', 'The planetarium show begins at 11:00 on weekdays.'),
 ('P06', 'Galleries', 'Dinosaur gallery', 'The dinosaur gallery is located on the second floor.'),
 ('P07', 'Galleries', 'Planetarium location', 'The planetarium is located on the first floor.'),
 ('P08', 'Workshops', 'Microscope workshop', 'The microscope workshop lasts 75 minutes.'),
 ('P09', 'Workshops', 'Chemistry demonstration', 'The chemistry demonstration lasts 45 minutes.'),
 ('P10', 'Opening', 'Weekly closure', 'The museum is closed on Mondays.'),
 ('P11', 'Opening', 'Midweek opening time', 'The museum opens at 10:00 from Tuesday to Friday.'),
 ('P12', 'Accessibility', 'Quiet room', 'The quiet room is beside the ground-floor reception desk.'),
 ('P13', 'Planetarium', 'Booking planetarium seats', 'Visitors must reserve planetarium seats online before arrival.'),
 ('P14', 'Events', 'Geology walk meeting point', 'The guided geology walk meets at the east garden gate.'),
 ('P15', 'Events', 'Sunday geology walk', 'The geology walk begins at 09:15 on Sundays.'),
 ('P16', 'Opening', 'Last entry', 'The final admission is at 16:00.'),
 ('P17', 'Opening', 'Weekday closing time', 'The museum closes at 17:00 on weekdays.'),
 ('P18', 'Events', 'Night sky session', 'The night sky session takes place on the roof terrace.')]

# Each entry has four parts: passage ID, section, title and complete text.
# The IDs are consecutive, which makes the source references easy to trace.
# A dictionary maps each ID (such as 'P01') to the passage and its metadata.
PASSAGES = {
    pid: {'id': pid, 'section': section, 'title': title,
          'text': text, 'source': 'Northbridge Visitor Guide'}
    for pid, section, title, text in ROWS
}

# Check the catalogue before building the index. These checks catch accidental
# duplicate, missing or out-of-order passage IDs early in the lab.
expected_ids = [f'P{i:02d}' for i in range(1, len(ROWS) + 1)]
actual_ids = [row[0] for row in ROWS]
assert actual_ids == expected_ids, 'Passage IDs must be consecutive from P01.'
assert len(PASSAGES) == len(ROWS), 'Each passage must have a unique ID.'

print(f'{len(PASSAGES)} passages loaded.')
print('Passage IDs:', ', '.join(PASSAGES))
print('First four passages:')
for pid, passage in list(PASSAGES.items())[:4]:
    print(f'[{pid}] {passage["section"]} | {passage["title"]}')
    print('   ', passage['text'])

# %% [markdown]
# ### Questions and reference answers
#
# To compare retrieval methods fairly, we use the **same questions** for both. A human-prepared evaluation set records the expected answer and the relevance of each passage.
#
# - Grade **3**: directly answers the question.
# - Grade **1**: related information, but not a complete answer.
# - Omitted passage: grade **0**, irrelevant.
#
# For Hit@k and Recall@k, only grade 3 is treated as relevant. The graded values also allow us to calculate nDCG. **The application must not look at these labels while answering**—they are used only afterwards for evaluation.

# %%
# Each labelled question stores a reference answer and relevance grades.
# These labels are used ONLY by the evaluation functions, never by ask().
EVAL = [{'id': 'Q01',
  'question': 'How much is adult entry on Saturdays?',
  'gold_answer': '£14',
  'grades': {'P01': 3, 'P02': 1}},
 {'id': 'Q02',
  'question': 'What is the weekday adult ticket price?',
  'gold_answer': '£11',
  'grades': {'P02': 3, 'P01': 1}},
 {'id': 'Q03',
  'question': 'When does the Saturday planetarium show start?',
  'gold_answer': '14:30',
  'grades': {'P04': 3, 'P05': 1}},
 {'id': 'Q04',
  'question': 'At what time is the planetarium performance during the working week?',
  'gold_answer': '11:00',
  'grades': {'P05': 3, 'P04': 1}},
 {'id': 'Q05',
  'question': 'Which floor has the dinosaur gallery?',
  'gold_answer': 'second floor',
  'grades': {'P06': 3, 'P07': 1}},
 {'id': 'Q06',
  'question': 'How long does the microscope workshop last?',
  'gold_answer': '75 minutes',
  'grades': {'P08': 3, 'P09': 1}},
 {'id': 'Q07', 'question': 'On which day is the museum shut?', 'gold_answer': 'Mondays', 'grades': {'P10': 3}},
 {'id': 'Q08',
  'question': 'How should visitors reserve seats for the planetarium?',
  'gold_answer': 'online',
  'grades': {'P13': 3}},
 {'id': 'Q09',
  'question': 'Where do participants meet for the geology walk?',
  'gold_answer': 'east garden gate',
  'grades': {'P14': 3, 'P15': 1}},
 {'id': 'Q10',
  'question': 'At what time does the geology walk start on Sunday?',
  'gold_answer': '09:15',
  'grades': {'P15': 3, 'P14': 1}},
 {'id': 'Q11',
  'question': 'Where is the quiet room?',
  'gold_answer': 'beside the ground-floor reception desk',
  'grades': {'P12': 3}},
 {'id': 'Q12',
  'question': 'What is the final entry time?',
  'gold_answer': '16:00',
  'grades': {'P16': 3, 'P17': 1}},
 {'id': 'Q13', 'question': "What is the museum cafe's telephone number?", 'gold_answer': '', 'grades': {}},
 {'id': 'Q14', 'question': 'What is the Wi-Fi password for visitors?', 'gold_answer': '', 'grades': {}}]
print(f'{len(EVAL)} evaluation questions prepared.')
print('Example:', EVAL[0]['question'])

# The source references should all point to passages in our museum guide.
assert all(set(item['grades']) <= set(PASSAGES) for item in EVAL)
assert [item['id'] for item in EVAL] == [f'Q{i:02d}' for i in range(1,len(EVAL)+1)]

# %% [markdown]
# ## Part 2 — Search with TF-IDF and BM25
#
# First we turn words into lowercase tokens and build an **inverted index**. An inverted index maps each term to the passage IDs containing it. We index the passage **title and text together**, while leaving the original text unchanged for displaying answers.
#
# We ignore a short list of common words in retrieval, such as *the* and *is*, because these are usually less useful for identifying the right passage.

# %%
def tokenize(text):
    """Step 1: Turn text into lowercase tokens (words or numbers)."""
    # Example: "The Planetarium!" becomes ["the", "planetarium"].
    return re.findall(r'[a-z0-9]+', text.lower())


# Common words often carry less information about the subject of a passage.
# Removing them is a simple search-preparation choice, not a general NLP rule.
STOPWORDS = {'a', 'an', 'and', 'are', 'at', 'by', 'can', 'do', 'does', 'for',
             'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'the', 'to',
             'was', 'were', 'what', 'when', 'where', 'which', 'with', 'you',
             'your', 'much', 'many', 'during', 'that'}


def search_tokens(text):
    """Step 2: Keep the tokens that will be used for search scoring."""
    return [word for word in tokenize(text) if word not in STOPWORDS]


# Step 3: Tokenise each passage's title AND text exactly once.
# We keep the original passage text separately for answering and printing.
TOKENS = {
    pid: search_tokens(passage['title'] + ' ' + passage['text'])
    for pid, passage in PASSAGES.items()
}

# Step 4: A Counter records the frequency of each term within a passage.
# Example: COUNTS['P01']['adult'] gives its term frequency in passage P01.
COUNTS = {pid: Counter(words) for pid, words in TOKENS.items()}

# Step 5: Build an inverted index: term -> IDs of passages containing it.
# set(words) ensures each passage is indexed once for each distinct term.
INDEX = {}
for pid, words in TOKENS.items():
    for word in set(words):
        INDEX.setdefault(word, set()).add(pid)

# Step 6: Document frequency DF[word] is the number of indexed passages
# containing the term. BM25 and TF-IDF use this to measure rarity.
DF = {word: len(passage_ids) for word, passage_ids in INDEX.items()}

# Step 7: BM25 also needs collection size and mean indexed passage length.
N = len(PASSAGES)
AVGDL = sum(len(words) for words in TOKENS.values()) / N

print('Index example:')
for word in ('planetarium', 'museum', 'discount'):
    print(f'{word:12s} -> {sorted(INDEX.get(word, set()))}')
print(f'Average indexed passage length: {AVGDL:.2f} tokens')

# %% [markdown]
# ### TF-IDF: count matching words and reward rare words
#
# The lecture defines **term frequency** and **inverse document frequency** as:
#
# $$
# \mathrm{tf}(t,d)=\mathrm{count}(t,d), \qquad
# \mathrm{idf}(t)=\log\!\left(\frac{N}{\mathrm{df}(t)}\right)
# $$
#
# The TF-IDF score for question $q$ and passage $d$ is:
#
# $$
# \mathrm{TFIDF}(q,d)=\sum_{t\in\mathrm{terms}(q)}
# \mathrm{tf}(t,d)\,\mathrm{idf}(t)
# $$
#
# Here $t$ is a query word, $N$ is the number of indexed passages, and $\mathrm{df}(t)$ is the number of passages containing that word. **In the code:** `COUNTS[pid][word]` supplies the term frequency; `DF[word]` supplies the document frequency; `math.log` calculates the natural logarithm.
#
# ### BM25: reduce the reward for repeated words and adjust for length
#
# The BM25 score is:
#
# $$
# \mathrm{BM25}(q,d)=\sum_{t\in\mathrm{terms}(q)}
# \mathrm{idf}_{\mathrm{BM25}}(t)
# \frac{f(t,d)(k_1+1)}
# {f(t,d)+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}
# $$
#
# Its IDF term is:
#
# $$
# \mathrm{idf}_{\mathrm{BM25}}(t)
# =\log\!\left(1+\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right)
# $$
#
# - $f(t,d)$: count of term $t$ in passage $d$.
# - $|d|$: number of indexed words in the passage; $\mathrm{avgdl}$: average indexed passage length.
# - $k_1=1.2$: controls how quickly the effect of repetition levels off.
# - $b=0.75$: controls the strength of length normalisation.
#
# **Watch the code:** `length_factor` implements the bracketed length term. Each matching query word adds one term to `score`. TF-IDF and BM25 use different score scales: **compare the order of passages**, not the size of their raw scores.

# %%
def tfidf_score(question, pid):
    """Add the TF-IDF contribution of each distinct query term."""
    # The score starts at 0; passages with no matching terms stay at 0.
    score = 0.0

    # Process each distinct query word once, matching the lecture's sum.
    for word in set(search_tokens(question)):
        if word not in DF:
            continue  # This word does not appear in any indexed passage.

        # TF = frequency of the query word in this passage.
        tf = COUNTS[pid][word]
        # IDF = natural log of (number of passages / document frequency).
        idf = math.log(N / DF[word])
        # Add this word's contribution to the passage score.
        score += tf * idf
    return score


def bm25_score(question, pid, k1=1.2, b=0.75):
    """Add the smoothed BM25 contribution of each distinct query term."""
    # The term in parentheses corrects for this passage's length.
    # Its value is 1 for a passage with average indexed length.
    length_factor = 1 - b + b * len(TOKENS[pid]) / AVGDL
    score = 0.0

    for word in set(search_tokens(question)):
        # f(t,d) = term frequency in this passage.
        frequency = COUNTS[pid][word]
        if frequency == 0:
            continue  # A query word absent from the passage adds nothing.

        # Smoothed IDF: rarer terms contribute more.
        idf = math.log(1 + (N - DF[word] + 0.5) / (DF[word] + 0.5))

        # The denominator causes diminishing returns for repeated words;
        # length_factor also adjusts for short/long passages.
        numerator = frequency * (k1 + 1)
        denominator = frequency + k1 * length_factor
        score += idf * numerator / denominator
    return score


def retrieve(question, method='bm25', section=None):
    """Return (passage ID, score) pairs sorted from highest to lowest."""
    # Step 1: Confirm that the requested method, question and filter are valid.
    if method not in ('tfidf', 'bm25'):
        raise ValueError("Method must be 'tfidf' or 'bm25'.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question must be nonempty text.')
    if section is not None and section not in {p['section'] for p in PASSAGES.values()}:
        raise ValueError('Unknown section; check the printed section names.')

    # Step 2: Choose one scoring function, without changing the document set.
    scorer = tfidf_score if method == 'tfidf' else bm25_score
    scores = []

    # Step 3: Score passages that pass any requested metadata filter.
    for pid, passage in PASSAGES.items():
        if section is not None and passage['section'] != section:
            continue
        score = scorer(question, pid)
        if score > 0:
            scores.append((pid, score))  # Do not present zero-score passages.

    # Step 4: Highest score first; resolve ties using the passage ID.
    return sorted(scores, key=lambda item: (-item[1], item[0]))


# Check one question with each retriever.
question = 'How much is adult entry on Saturdays?'
for method in ('tfidf', 'bm25'):
    print(f'\n{method.upper()} ranking for: {question}')
    for pid, score in retrieve(question, method)[:3]:
        print(f'  [{pid}] score={score:.3f} | {PASSAGES[pid]["title"]}')

# %% [markdown]
# ## Part 3 — Turn retrieved passages into an answer
#
# A high retrieval score tells us that a passage seems **relevant**, not that it contains a complete answer. The answer extractor therefore checks the retrieved text separately.
#
# Our museum questions usually ask for a **price**, **time**, **duration**, **floor**, **day**, **location**, or **booking method**. The extractor first identifies the requested kind of answer and then searches the passage for a matching text pattern.
#
# For example, a question beginning *How much ...?* calls for a price. The extractor looks for an amount such as `£14` **inside the passage itself**. When nothing suitable appears, it returns an empty string. This is a rule-based extractor; it does not predict start/end probabilities.
#
# The function `compatible_evidence()` performs an additional check so that a weekday notice is not used for a Saturday question. This illustrates why finding a word such as *planetarium* is not always sufficient to answer the question.

# %%
def answer_type(question):
    """Step 1: Identify the kind of short answer the visitor wants."""
    q = question.lower()
    # Test the more specific forms first; 'when' and 'where' are general.
    if re.search(r'how much|price|cost|discount', q):
        return 'price'
    if re.search(r'how long|duration', q):
        return 'duration'
    if re.search(r'which floor|what floor', q):
        return 'floor'
    if re.search(r'which day|what day|on which day', q):
        return 'day'
    if re.search(r'where|meeting point', q):
        return 'location'
    if re.search(r'how.*(book|reserve)|reserve seats', q):
        return 'booking'
    if re.search(r'what time|at what time|when|entry time|opening time', q):
        return 'time'
    return 'unsupported'


def extract_answer(question, passage_text):
    """Step 2: Return a phrase copied from the passage, or empty text."""
    kind = answer_type(question)

    # Each regular expression describes a recognisable answer form.
    # These are simple patterns: they do not evaluate the truth of a sentence.
    patterns = {
        'price': r'£\s*\d+(?:\.\d{1,2})?',
        'duration': r'\b\d+\s+(?:minutes?|hours?)\b',
        'floor': r'\b(?:ground|first|second|third)\s*-?\s*floor\b',
        'day': r'\b(?:Mondays?|Tuesdays?|Wednesdays?|Thursdays?|Fridays?|Saturdays?|Sundays?)\b',
        'time': r'\b(?:[01]?\d|2[0-3]):[0-5]\d\b',
        'booking': r'\bonline\b',
    }

    # Locations are phrases rather than fixed formats such as HH:MM.
    if kind == 'location':
        match = re.search(
            r'\b(?:is|are|meets|takes place)\s+(at|on|in|near|beside)\s+([^.!?]+)',
            passage_text, flags=re.I)
        if not match:
            return ''
        place = match.group(2).strip()
        if match.group(1).lower() in ('beside', 'near'):
            return match.group(1) + ' ' + place
        return re.sub(r'^the\s+', '', place, flags=re.I)

    # If the type is unsupported, or the passage has no matching pattern,
    # return an empty answer. Do not invent a value.
    if kind not in patterns:
        return ''
    match = re.search(patterns[kind], passage_text, flags=re.I)
    return match.group(0) if match else ''


def compatible_evidence(question, passage):
    """Step 3: Reject obviously mismatched topics or days."""
    q = set(tokenize(question))
    words = set(tokenize(passage['title'] + ' ' + passage['text']))

    # A dinosaur question should not be answered using a different gallery.
    for topic in ('planetarium', 'microscope', 'dinosaur', 'geology', 'quiet', 'student', 'adult'):
        if topic in q and topic not in words:
            return False

    # Compare the requested day with the day in the retrieved notice.
    if q & {'saturday', 'saturdays'} and words & {'weekday', 'weekdays', 'sundays'}:
        return False
    if q & {'weekday', 'weekdays', 'working'} and words & {'saturday', 'saturdays', 'sundays'}:
        return False
    if q & {'sunday', 'sundays'} and words & {'saturday', 'saturdays', 'weekday', 'weekdays'}:
        return False
    return True


print('Answer extraction functions ready.')

# %% [markdown]
# ### Follow a question through the complete pipeline
#
# There are two functions with different purposes:
#
# - **`ask(question, ...)`** performs the computation and **returns a dictionary**. It does not print anything. That makes it suitable for repeated calls during evaluation.
# - **`show_answer(question, ...)`** calls `ask()`, **prints the results in full** and returns the same dictionary so you can inspect it afterwards.
#
# The pipeline is:
#
# ```text
# question
#    |--> retrieve(question) --> ranked (passage ID, score) pairs
#    |                              |
#    |                              v
#    |                         keep the first k
#    |                              |
#    '-----> match question type -> check each passage -> extract phrase
#                                                   |
#                                                   v
#                                          answer + supporting source
# ```
#
# **Reading `show_answer()` step by step:** (1) call `ask()`; (2) print the question and method; (3) loop through `result['shown']` and print every retrieved passage without shortening it; (4) print the answer and source; (5) return `result`. The printed **extraction checks** explain why each passage was accepted or skipped. The reference answers in `EVAL` are never used to produce the answer.

# %%
def ask(question, method='bm25', top_k=3, section=None):
    """Compute one answer and return the evidence and decisions as a dictionary."""
    # Step 1: Validate how many passages the caller wants to inspect.
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive whole number.')

    # Step 2: Run the selected retriever. This is a full ranked list.
    ranking = retrieve(question, method, section)

    # Step 3: Keep at most k passages. These will also be printed later.
    shown = ranking[:top_k]
    answer, source = '', None
    checks = []  # Notes explaining what happened at each passage.

    # Step 4: Examine passages in ranked order, from best score downwards.
    for pid, score in shown:
        passage = PASSAGES[pid]

        # A high search score is not enough if the topic/day is wrong.
        if not compatible_evidence(question, passage):
            checks.append(f'[{pid}] rejected: topic or day does not match')
            continue  # Try the next retrieved passage.

        # Look for a short answer of the requested type in this passage.
        candidate = extract_answer(question, passage['text'])
        if candidate:
            answer, source = candidate, pid
            checks.append(f'[{pid}] extracted: {candidate}')
            break  # Stop at the first acceptable answer.
        checks.append(f'[{pid}] no matching answer phrase')

    # Step 5: Keep both the complete ranking and inspected top-k passages.
    # The reference answers are NOT used here: they are for evaluation only.
    return {
        'question': question, 'method': method, 'section': section,
        'ranking': ranking, 'shown': shown, 'answer': answer,
        'source': source, 'checks': checks,
    }


def show_answer(question, method='bm25', top_k=3, section=None):
    """Print the complete retrieval and answer report; also return the result."""
    # Step 1: Ask the system to compute the result without printing.
    result = ask(question, method, top_k, section)

    # Step 2: Print the visitor's question and chosen retrieval settings.
    print('=' * 65)
    print('QUESTION:', result['question'])
    print('RETRIEVER:', method, '| TOP-K:', top_k,
          '| SECTION:', section if section else 'all')

    # Step 3: Print ALL selected passages, including their entire text.
    # We use result['shown'] rather than result['ranking'] because top_k
    # controls how many passages the answer extractor was allowed to see.
    print('\nRETRIEVED PASSAGES (full text):')
    if not result['shown']:
        print('  No matching passages.')
    for position, (pid, score) in enumerate(result['shown'], start=1):
        passage = PASSAGES[pid]
        print(f'  {position}. [{pid}] {passage["title"]} '
              f'({passage["section"]}) | score={score:.3f}')
        print('     ' + passage['text'])  # No slicing, abbreviation or truncation.

    # Step 4: Display the extracted phrase, source and decision trace.
    print('\nANSWER:', result['answer'] or '[No supported answer found]')
    if result['source']:
        pid = result['source']
        print('SOURCE:', pid, '-', PASSAGES[pid]['source'])
    print('EXTRACTION CHECKS:', '; '.join(result['checks']) or 'No evidence to examine.')
    print('=' * 65 + '\n')

    # Step 5: Return the dictionary. A later cell can access result['answer'],
    # result['shown'], etc., without reading the printed output.
    return result


show_answer('Where is the quiet room?')

# %% [markdown]
# ### Try the museum guide
#
# The following questions deliberately contain competing passages—for instance, weekday versus Saturday prices and show times. Check that the **answer and the source** agree with the question, rather than trusting the score alone.

# %%
show_answer('How much is adult entry on Saturdays?', method='bm25', top_k=3)
show_answer('When does the Saturday planetarium show start?', method='tfidf', top_k=3)
show_answer('What is the museum cafe telephone number?', method='bm25', top_k=3)

# %% [markdown]
# ## Part 4 — Evaluate retrieval and answers separately
#
# A retrieval system can retrieve the right passage but still extract the wrong words. We therefore use two groups of metrics:
#
# | Retrieval metric | What it tells us |
# |---|---|
# | **Hit@k** | Was at least one relevant passage found in the top $k$? |
# | **Recall@k** | What fraction of all relevant passages was found in the top $k$? |
# | **MRR** | How early does the *first* relevant passage appear? |
# | **nDCG@k** | Are highly relevant passages ranked near the top? |
#
# | Answer metric | What it tells us |
# |---|---|
# | **Exact Match (EM)** | Is the predicted answer exactly the same as the reference after normalisation? |
# | **Token F1** | How much token overlap is there between the two answers? |
#
# For a single question $q$, **Hit@k** is either 1 or 0. **Recall@k** is the number of relevant passages found in the first $k$, divided by the total number of relevant passages for that question. For **MRR**, let $r$ be the rank of the first relevant passage; its reciprocal rank is $1/r$ (or 0 if no relevant passage is ranked).
#
# **nDCG@k** uses graded relevance. First calculate the discounted gain of the returned ranking:
#
# $$
# \mathrm{DCG@}k=\sum_{i=1}^{k}
# \frac{2^{\mathrm{rel}_i}-1}{\log_2(i+1)}
# $$
#
# Then compare this with the gain of the ideal ranking, **IDCG@k**:
#
# $$
# \mathrm{nDCG@}k=\frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}
# $$
#
# Here $\mathrm{rel}_i$ is the relevance grade at rank $i$, and $\log_2$ is the base-2 logarithm. In our labels, **3** means directly answers the question, **1** means related but incomplete, and **0** means irrelevant. Only grade 3 counts as *relevant* for Hit@k, Recall@k and MRR; nDCG uses all grades.
#
# For the answer metrics, let $M$ be the number of matching token occurrences, $N_{\mathrm{pred}}$ the predicted-answer length, and $N_{\mathrm{ref}}$ the reference-answer length:
#
# $$
# P=\frac{M}{N_{\mathrm{pred}}}, \qquad
# R=\frac{M}{N_{\mathrm{ref}}}, \qquad
# \mathrm{F1}=\frac{2PR}{P+R}
# $$
#
# The functions below handle empty answers separately, and all **dataset-level metrics are averages across questions**. Questions with no relevant passages are excluded from retrieval averages; they still count in answer evaluation. The answer functions do not consult these reference labels.

# %%
def normalise(text):
    """Prepare a string for answer comparison: lowercase and remove punctuation."""
    # Articles such as 'a' and 'the' are kept, matching the lecture example.
    return ' '.join(tokenize(text))


def exact_match(predicted, reference):
    """Return 1 if the normalised answers match exactly, otherwise 0."""
    return int(normalise(predicted) == normalise(reference))


def token_f1(predicted, reference):
    """Give partial credit based on shared word occurrences."""
    # Counter keeps repeated words and therefore avoids overcounting matches.
    pred = Counter(tokenize(predicted))
    gold = Counter(tokenize(reference))

    # Handle two empty answers and one empty answer explicitly.
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    # Counter intersection keeps the minimum occurrence count per token.
    matched = sum((pred & gold).values())
    if matched == 0:
        return 0.0

    # Precision: fraction of predicted tokens found in the reference.
    # Recall: fraction of reference tokens found in the prediction.
    precision = matched / sum(pred.values())
    recall = matched / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def retrieval_metrics(ranking, grades, k):
    """Evaluate one question's ranking using reference relevance grades."""
    # A grade of 3 means a passage directly answers the question.
    # Grade 1 is partially related but is not a complete answer.
    relevant = {pid for pid, grade in grades.items() if grade >= 2}
    if not relevant:
        return None  # Unanswerable questions have no target retrieval set.

    ranked_ids = [pid for pid, score in ranking]

    # Hit@k: is there at least one relevant passage among the first k?
    # Recall@k: how many of the relevant passages have been found?
    found = set(ranked_ids[:k]) & relevant
    hit = float(bool(found))
    recall_at_k = len(found) / len(relevant)

    # Reciprocal Rank: take only the FIRST relevant passage in the ranking.
    first_rank = next(
        (rank for rank, pid in enumerate(ranked_ids, start=1)
         if pid in relevant),
        None,
    )
    rr = 1 / first_rank if first_rank else 0.0

    # DCG: a relevance grade at rank i contributes
    # (2**grade - 1) / log2(i + 1). Earlier positions carry more weight.
    dcg = sum(
        (2 ** grades.get(pid, 0) - 1) / math.log2(rank + 1)
        for rank, pid in enumerate(ranked_ids[:k], start=1)
    )

    # IDCG: sort ALL known relevance grades from highest to lowest.
    # This is the best achievable ordering, not just a rearrangement
    # of the documents that our retriever happened to find.
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    ndcg = dcg / idcg if idcg else 0.0

    return {'hit': hit, 'recall': recall_at_k, 'rr': rr, 'ndcg': ndcg}


# Quick checks make it easier to catch mistakes during editing.
assert exact_match('£14', '£14') == 1
assert exact_match('a short answer', 'short answer') == 0
assert token_f1('short demo description for the showcase',
                'a short demo description') == 0.6
print('Evaluation metric checks passed.')

# %%
def mean(values):
    """Compute the ordinary arithmetic average (or 0 for an empty list)."""
    return sum(values) / len(values) if values else 0.0


def evaluate(method='bm25', k=3):
    """Run every question, then compare predictions with reference labels."""
    retrieval_scores, answer_scores, detail = [], [], []

    for item in EVAL:
        # Step 1: Make a prediction using the selected retrieval method.
        result = ask(item['question'], method=method, top_k=k)

        # Step 2: Measure retrieval only when relevant passages exist.
        ranking_score = retrieval_metrics(result['ranking'], item['grades'], k)
        if ranking_score is not None:
            retrieval_scores.append(ranking_score)

        # Step 3: Measure the predicted answer against the reference answer.
        # For an unanswerable question the reference is an empty string.
        answer_scores.append((
            exact_match(result['answer'], item['gold_answer']),
            token_f1(result['answer'], item['gold_answer']),
        ))
        detail.append((item, result))  # Helpful for diagnosing failures.

    # Step 4: Average the per-question results for each metric.
    return {
        'Hit': mean([s['hit'] for s in retrieval_scores]),
        'Recall': mean([s['recall'] for s in retrieval_scores]),
        'MRR': mean([s['rr'] for s in retrieval_scores]),
        'nDCG': mean([s['ndcg'] for s in retrieval_scores]),
        'EM': mean([s[0] for s in answer_scores]),
        'F1': mean([s[1] for s in answer_scores]),
        'details': detail,
    }


def compare_methods(k=3):
    """Print two rows using the SAME questions and evaluation rules."""
    print(f'Comparison using top-{k} passages')
    print(f'{"Method":9s}  {"Hit@k":>7s} {"Recall@k":>9s} {"MRR":>7s} '
          f'{"nDCG@k":>8s} {"EM":>7s} {"F1":>7s}')
    print('-' * 65)

    for method in ('tfidf', 'bm25'):
        scores = evaluate(method, k)
        print(f'{method:9s}  {scores["Hit"]:7.3f} {scores["Recall"]:9.3f} '
              f'{scores["MRR"]:7.3f} {scores["nDCG"]:8.3f} '
              f'{scores["EM"]:7.3f} {scores["F1"]:7.3f}')

    # Retrieval excludes questions with no relevant source; EM/F1 include them.
    answerable = sum(bool(item['grades']) for item in EVAL)
    print(f'Retrieval: {answerable} answerable questions; answers: {len(EVAL)} total.')


compare_methods(k=3)

# %% [markdown]
# ## Part 5 — Your investigation
#
# The application is complete. These **four short tasks** ask you to change one setting at a time, inspect the complete passages, and make a judgement based on the evidence. Each code cell runs as supplied; you can then modify the values to carry out your own comparisons.

# %% [markdown]
# ### Task 1. Does the retrieval method change the answer?
#
# Try the same admission question with TF-IDF and BM25. Compare the **first-ranked passage**, the extracted answer and its source. Then change the question to ask about weekday admission.

# %% [markdown]
# **Example interpretation (solution):** Both methods should be compared using the same question and evidence. They may rank identical passages differently because BM25 adjusts for frequency saturation and passage length. A higher retrieval score does not automatically mean a correct answer.

# %%
TASK_QUESTION = 'How much is adult entry on Saturdays?'
for method in ('tfidf', 'bm25'):
    show_answer(TASK_QUESTION, method=method, top_k=2)

# %% [markdown]
# ### Task 2. Filter by section
#
# Ask a planetarium question with and without a `section='Planetarium'` filter. Which passages disappear? Why might a restrictive filter be dangerous if the selected section is wrong?

# %% [markdown]
# **Example interpretation (solution):** The section filter removes every passage outside Planetarium before scoring. This can reduce noise, but if the true evidence is catalogued in another section, the answer becomes impossible to retrieve.

# %%
FILTER_QUESTION = 'When does the Saturday planetarium show start?'
show_answer(FILTER_QUESTION, method='bm25', top_k=3)
show_answer(FILTER_QUESTION, method='bm25', top_k=3, section='Planetarium')

# %% [markdown]
# ### Task 3. How much evidence is enough?
#
# Compare $k=1$ and $k=3$. The extractor may inspect more passages at larger $k$. Use the benchmark to check Hit@k and answer EM, then inspect an individual question where the result changes (if any).

# %% [markdown]
# **Example interpretation (solution):** Hit@k and Recall@k cannot decrease when k increases for a fixed ranking. Exact Match can behave differently, because the rule-based extractor may select a plausible phrase from the wrong passage.

# %%
for k in (1, 3):
    s = evaluate('bm25', k)
    print(f'BM25 top-{k}: Hit@{k}={s["Hit"]:.3f} | '
          f'Recall@{k}={s["Recall"]:.3f} | EM={s["EM"]:.3f}')

# %% [markdown]
# ### Task 4. Diagnose an incorrect or missing answer
#
# Choose a question from the evaluation set and compare the prediction with the reference answer. If they differ, decide whether the **relevant passage was missing from the top $k$** or **the extractor failed despite having the evidence**. Also try an unanswerable question; a sensible system should not invent facts.

# %% [markdown]
# **Example interpretation (solution):** Check whether the gold source ID is present among the printed passages. If not, it is a retrieval issue. If present but the answer is wrong or absent, the extractor has a limitation. For a question with no answer in the source, the empty prediction is preferable to an invented answer.

# %%
# Pick a question number between 0 and 13 to investigate.
CASE_NUMBER = 3
item = EVAL[CASE_NUMBER]
result = show_answer(item['question'], method='bm25', top_k=3)
print('REFERENCE ANSWER:', item['gold_answer'] or '[No answer in the guide]')
print('RELEVANT SOURCES:', [pid for pid, grade in item['grades'].items() if grade >= 2])
print('EXACT MATCH:', exact_match(result['answer'], item['gold_answer']))

show_answer('What is the museum cafe telephone number?', method='bm25', top_k=3)

# %% [markdown]
# ## Wrap-up
#
# In about **100 words**, explain which retrieval method you would choose for this small museum guide. Refer to at least one retrieval metric, one answer metric, and one actual passage you inspected. A good explanation recognises that the retrieval scores may improve without the simple answer extractor necessarily improving.
#
# **Remember:** TF-IDF and BM25 are real lexical scoring functions implemented here. The answer extractor is a deliberately small rule-based teaching example; it does not implement the trained neural start/end predictions described in the lecture.

