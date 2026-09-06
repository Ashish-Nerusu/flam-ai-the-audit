# Chronological Lab Notebook

## Part A: Tokenizer Audit

### 1. Macro-average vs Micro-average
**Hypothesis:** `fertility.py` uses a macro-average (averaging the per-line ratios) which skews the metric compared to a corpus-level micro-average (sum of all tokens / sum of all words).
**Command:** `python test_bugs.py` (Script created to run the original `analyze_v0` and a new `analyze_v1_micro` on the starter corpus).
**Result:** 
- Original (Macro Average): ENG 1.27, HIN 7.45
- Micro Average: ENG 1.25, HIN 7.40
**Interpretation:** The hypothesis is supported. The original script overestimates fertility slightly due to weighting short outlier lines identically to long lines.

### 2. Exploring `.lower()` and NFC normalization
**Hypothesis:** The assignment hints that something looks suspicious but is actually fine. I suspected either `line.lower()` or `unicodedata.normalize("NFC", line)`.
**Command (NFC):** `python -c "import tiktoken, unicodedata; encode = tiktoken.get_encoding('gpt2').encode; lines = [l.strip() for l in open('partA/corpus/starter_hin.txt', encoding='utf-8') if l.strip()]; print(sum(len(encode(l)) for l in lines)); print(sum(len(encode(unicodedata.normalize('NFC', l))) for l in lines))"`
**Result (NFC):** Both outputs were `459`. 
**Command (.lower):** Run `test_bugs.py` with `.lower()` bypassed.
**Result (.lower):** English changes from 1.25 to 1.23. Hindi remains 7.45.
**Interpretation:** NFC has 0 effect on token counts in our Hindi sample, and `.lower()` has a negligible impact on English. These are the "suspicious but harmless" elements.

### 3. Word Counting Logic (`split(" ")`)
**Hypothesis:** `line.split(" ")` mishandles multiple spaces, counting empty strings as words and deflating fertility.
**Command:** Evaluated double spaces in Python: `python -c "print(len('books  in'.split(' ')))"` -> Output: `3`.
Then ran `test_split_distortion.py` to compare `split(" ")` vs `split()`.
**Result:** 
- `split(" ")`: ENG fertility 1.265, HIN fertility 7.448.
- `split()`: ENG fertility 1.283, HIN fertility 7.598.
**Interpretation:** The double-space bug artificially inflates the word count, causing the original script to underestimate fertility for both languages. Tok/char remained identical (0.226 and 1.579).

### 4. Conceptual Denominator Flaw
**Hypothesis:** Comparing languages via "tokens per word" obscures reality because Hindi uses fewer words to express the same meaning. For this benchmark, the parallel sentence is the most defensible denominator because it approximately holds translated content constant across languages.
**Command:** `python test_sentence.py` (sums total words and tokens across the parallel corpora).
**Result:** English uses 78 words and 96 tokens. Hindi uses 61 words and 459 tokens. The Hindi/English token ratio for this parallel sample is 459/96 = 4.78x.
**Interpretation:** The previous report's claim of a 6-7x token workload multiple is sensitive to the choice of denominator. The parallel-sentence calculation gives 4.78x on this starter sample. A metric of 4.78x per parallel meaning demonstrates severe denominator sensitivity.

### 5. Cross-Language Recomputation (A3)
**Hypothesis:** An Indic-aware tokenizer will normalize the token ratio penalty for Dravidian languages compared to GPT-2.
**Experiment:** Attempted to use the HuggingFace `datasets` library to pull FLORES-200. Encountered `Dataset scripts are no longer supported` errors, followed by a Windows `charmap` codec error in the dataset script.
**Revision:** Downgraded `datasets<3` and used `Muennighoff/flores200` with `trust_remote_code=True` and `PYTHONUTF8=1`.
**Command:** `python prepare_corpus.py` (downloaded exactly 500 lines for `eng_Latn`, `hin_Deva`, `kan_Knda`, `tel_Telu`).
**Command:** `python corrected_analysis.py`
**Result:** 
- gpt2 Tok/Sentence: Eng 26.0, Hin 187.8, Kan 343.9, Tel 327.4
- xlm-roberta Tok/Sentence: Eng 29.5, Hin 36.6, Kan 40.2, Tel 38.6
**Interpretation:** `gpt2` requires 13.21x (343.9/26.0) more tokens for Kannada compared to English. `xlm-roberta` brings this ratio down to 1.36x (40.2/29.5).

---

## Part B: Capacity Reconciliation

### 1. Theoretical KV Cache Capacity
**Hypothesis:** We can derive the KV bytes/token exactly from the model spec and estimate the maximum sequence capacity under the stated memory assumptions.
**Arithmetic & Command:** `python -c "bytes_per_token = 2 * 28 * 8 * 128 * 2; print(bytes_per_token)"` -> `114688`. 
`114688 * 4096 = 469,762,048` bytes per sequence.
Usable memory = (24 * 10**9 * 0.92) - (4.2 * 10**9 * 2) - (1.6 * 10**9) = 12,080,000,000 bytes.
Capacity = 12,080,000,000 / 469,762,048 = 25.7 sequences.
**Interpretation:** This matches row 12 of `bench_log.csv` where 24 sequences use 93% of the cache (`24/0.93 = 25.8`).

### 2. Deriving Goodput
**Hypothesis:** `reported_tok_s` includes prompt tokens, masking the poor generation throughput of long contexts.
**Command:** `python -c "print((24 * 512) / 61.16); print(1607.4 * (512 / (3584 + 512)))"`
**Result:** Both outputs are exactly `200.91`.
**Interpretation:** The intern misread the column. Honest goodput drops to 201 tok/s at batch 24, much lower than the 294 tok/s for short prompts.
