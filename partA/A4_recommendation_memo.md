# Part A Recommendation Memo

## Corrected Headline Numbers
Our previous analysis overstated the relative token workload of serving Indic languages by using a metric highly sensitive to language-specific morphology (tokens per word). Because different languages express the same meaning using different numbers of words and characters, the parallel sentence is the most defensible denominator for this benchmark because it approximately holds translated content constant across languages.

When measured correctly on a 500-sentence multilingual parallel corpus (FLORES-200), the token ratio (relative to English) is:

| Language | GPT-2 Ratio | XLM-RoBERTa Ratio |
|----------|-------------|-------------------|
| Hindi    | 7.21x       | 1.24x             |
| Kannada  | 13.21x      | 1.36x             |
| Telugu   | 12.58x      | 1.31x             |

On the 500-sentence FLORES-200 parallel benchmark, GPT-2 produced 13.21× as many tokens per sentence for Kannada versus English, while XLM-R produced 1.36×. This indicates that tokenizer choice is a major determinant of multilingual tokenization overhead.

## Routing Recommendation
For routing, I would use measured tokens per equivalent parallel request as the planning metric and monitor actual generated tokens/request in production. Route all Indic traffic to models using a multilingual/Indic-aware tokenizer (e.g., XLM-RoBERTa or a specialized Indic tokenizer) to avoid the massive tokenization overhead of GPT-2.

## Biggest Caveat
The eval corpus is drawn entirely from FLORES-200 (translated Wikipedia sentences), which is formal and clean. It does not reflect the informal, code-mixed, or typo-ridden text (e.g., Hinglish, transliteration) typical of real user queries. If users heavily use Romanized Indic languages rather than native scripts, the tokenization behavior of both GPT-2 and XLM-R will differ from this benchmark.

## Production Metric to Monitor
**Average generated tokens per completed user request, grouped by language**. If the real-world tokens/request for Kannada is dramatically higher than our 1.36x benchmark expectation, it indicates our synthetic Wikipedia corpus failed to capture real usage patterns (e.g., users being highly conversational, or the model becoming unnaturally verbose).
