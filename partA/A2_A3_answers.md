# A2: Audit the Script and the Metric

We audited `fertility.py` and identified two implementation/metric issues and one major conceptual problem. 

### 1. Aggregation mismatch: macro-averaged per-line fertility is not corpus-level tokens/word
`fertility.py` calculates the average tokens/word by taking the ratio per line and averaging those ratios (a macro-average), rather than summing all tokens and dividing by all words (a micro-average). 
* **Command:** `python test_bugs.py --eng corpus/starter_eng.txt --hin corpus/starter_hin.txt`
* **Result:** Macro-averaging yields 7.45 tok/word for Hindi and 1.27 for English. The corrected corpus-level aggregation yields 7.40 and 1.25 respectively. 
* **Distortion:** It slightly overstates the fertility metric (by 0.05 for Hindi and 0.02 for English) because it mathematically weights short, outlier sentences equally with long sentences.

### 2. Implementation Bug: Naive Word Counting (`split(" ")`)
The script uses `line.split(" ")` to count words. This fails to handle multiple consecutive spaces properly, producing empty strings that artificially inflate the word count.
* **Command:** `python test_split_distortion.py --eng corpus/starter_eng.txt --hin corpus/starter_hin.txt`
* **Result Before (`split(" ")`):** ENG fertility: 1.265, tok/char: 0.226. HIN fertility: 7.448, tok/char: 1.579.
* **Result After (`split()`):** ENG fertility: 1.283, tok/char: 0.226. HIN fertility: 7.598, tok/char: 1.579.
* **Distortion:** By inflating the denominator with empty strings, the original script underestimates fertility for English by ~0.018 and Hindi by ~0.150. (As expected, tok/char remains identical).

### 3. Suspicious-but-valid: `unicodedata.normalize("NFC", line)`
* NFC normalization is the suspicious-but-valid operation I tested. On the Hindi starter corpus it changed token count from 459 to 459, so I found no evidence of distortion. (Separately, `.lower()` is a preprocessing choice that changes English tokenization slightly, so it is not completely harmless).

### 4. Conceptual Denominator Problem
The report uses "tokens per word" and "tokens per character" to compare cross-language token ratios. This is conceptually flawed. Different languages express the same meaning using different numbers of words and characters. Comparing them directly using these denominators heavily penalizes languages with complex scripts or denser morphology.
* **Command:** `python test_sentence.py --eng corpus/starter_eng.txt --hin corpus/starter_hin.txt`
* **Result:** In the `corpus_sample`, English uses 78 words and 96 GPT-2 tokens, while the exact same meaning in Hindi uses 61 words and 459 GPT-2 tokens.
* **Interpretation:** The Hindi/English GPT-2 token ratio on this specific parallel starter sample is 4.78x (459/96). The previous report claimed a 6x to 7x multiple. This demonstrates severe denominator sensitivity. For this benchmark, parallel sentence is the most defensible denominator because it holds the translated content approximately constant across languages.

---

# A3: Corrected Analysis

We ran an evaluation on a robust parallel corpus to recompute cross-language metrics.

### Corpus Details (A1)
* **Source:** `Muennighoff/flores200` (a mirror of `facebook/flores` dev split)
* **Size:** Exactly 500 parallel sentences per language.
* **Language Codes:** `eng_Latn` (English), `hin_Deva` (Hindi), `kan_Knda` (Kannada), `tel_Telu` (Telugu).
* **Domain:** Translated Wikipedia sentences. Limitation: Does not reflect informal, code-mixed, or typo-ridden text typical of real user queries.
* **Preprocessing:** Each FLORES sentence was decoded as UTF-8 and stripped of leading/trailing whitespace. No lowercasing or Unicode normalization was applied during the corrected A3 analysis so that tokenization reflects the supplied text.

### Methodology
* **Tokenizers:** `gpt2` (tiktoken v0.14.0) and `hf:xlm-roberta-base` (transformers v5.16.1).
* **Command:** `python corrected_analysis.py`
* **Denominators measured:** tokens per whitespace word, tokens per UTF-8 byte, and tokens per parallel sentence.

### Raw Results
**Tokenizer: gpt2**
| Language | Tok/Word | Tok/Byte | Tok/Sentence |
|----------|----------|----------|--------------|
| eng      | 1.24     | 0.209    | 26.0         |
| hin      | 7.77     | 0.593    | 187.8        |
| kan      | 22.31    | 0.977    | 343.9        |
| tel      | 20.31    | 0.988    | 327.4        |

**Tokenizer: xlm-roberta**
| Language | Tok/Word | Tok/Byte | Tok/Sentence |
|----------|----------|----------|--------------|
| eng      | 1.41     | 0.236    | 29.5         |
| hin      | 1.52     | 0.116    | 36.6         |
| kan      | 2.61     | 0.114    | 40.2         |
| tel      | 2.40     | 0.117    | 38.6         |

### Which single number should drive a routing decision?
**Tokens per Parallel Sentence**. 
When a user asks a question, the assistant generates an answer conveying a specific amount of information (meaning). Since parallel sentences hold meaning constant, the ratio of tokens per parallel sentence best represents the relative compute workload of serving that language. 

**Calculation behind the multipliers:**
We divide the Tok/Sentence of each language by the English reference. 
* Kannada (gpt2): `343.9 / 26.0 = 13.21x`
* Kannada (xlm-roberta): `40.2 / 29.5 = 1.36x`

Using the legacy GPT-2 tokenizer forces Kannada and Telugu to use over 13x more tokens than English for the same meaning. The XLM-RoBERTa tokenizer brings this ratio down to ~1.3x.
