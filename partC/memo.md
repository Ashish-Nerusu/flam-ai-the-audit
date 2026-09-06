# Decision Memo: Casualization in Indic Languages

**Recommendation: Path (a) — LoRA/PEFT SFT on synthetic "casualized" response pairs.**

Prompt engineering is the cheapest baseline, but it may require increasingly complex prompts and does not guarantee consistent style behavior across all six languages. I would therefore test it as a baseline while prioritizing LoRA SFT if the synthetic-data viability experiment succeeds. An inference-time rewriter (b) would add a second model to the pipeline, increasing serving latency and compute overhead. I would use parameter-efficient SFT (LoRA) rather than full fine-tuning to reduce memory and training risk within the two-week compute budget, incorporating the casual style into the model.

### Quantities & Arithmetic

* **Reviewer Capacity [Known Constraint]:** 10 hours/week × 2 weeks = 20 hours before the Week 3 launch review.
* **Reviewer Speed [Assumption]:** 1 minute per response pair. 
* **Evaluation Size [Estimate]:** 300 synthetic response pairs (150 Hindi, 150 Kannada) held out for the final blind A/B test. This consumes 300 minutes (5 hours) of reviewer time.
* **Reviewer Training Capacity [Calculated]:** 20 hours - 5 hours = 15 hours. At 1 min/pair, the reviewer can verify/correct ~900 highly-casual Hindi and Kannada training pairs. (The 900 reviewer-verified pairs are Hindi/Kannada only, because those are the languages the reviewer can directly evaluate).
* **Synthetic Dataset Size [Assumption]:** 10,000 synthetic response pairs across all 6 languages (including the 900 reviewer-verified pairs).
* **Estimated Tokens [Calculated]:** 10,000 pairs × ~500 tokens/pair = ~5 million tokens.
* **Generation Workload [Estimate]:** Few-shot prompting the 4B model to generate 9,100 unverified pairs. At an assumed 200 tok/s goodput on an A100, this takes `(9,100 * 500) / 200 = 22,750 seconds` ≈ 6.3 hours.
* **Training Workload [Estimate]:** Training tokens = 5M * 3 epochs = 15M tokens. Assumption: 500-2,000 effective training tokens/sec on one A100 for the planned LoRA configuration. `15M / 2000 = 2.1 hours`. `15M / 500 = 8.3 hours`. Under this planning assumption, training would take approximately 2-8 hours, leaving substantial room in the two-week window. This is an estimate and would be replaced by a measured throughput benchmark during the first training run.
* **Serving Architecture [Reasoned]:** No additional model stage is introduced at inference time, unlike the rewriter option. Therefore the architectural serving overhead is approximately zero; actual end-to-end latency/token usage must still be benchmarked after fine-tuning.

### Success Metric
**A blind pairwise preference evaluation.**
* **Evaluation Set:** 300 held-out prompts.
* **Languages:** Hindi + Kannada (because these are the languages the native reviewer can evaluate). Tamil/Telugu/Bengali/Marathi require automated/LLM-based proxy evaluation initially.
* **Each prompt:** original model vs SFT output.
* **Success Threshold:** SFT is preferred in **≥70%** of valid comparisons by the native reviewer.

### Primary Kill Criterion
**Style Transfer Failure.** By Day 5, if the SFT pilot does not achieve at least 60% preference over the original model on a 100-example Hindi/Kannada reviewer set, abandon the SFT path and move to prompt/rewrite experimentation.

### Safety Guardrail
Reject the synthetic-data pipeline if the sampled outputs show an unacceptable safety/error rate. Toxicity/hallucination is a safety constraint, evaluated by the reviewer during data sampling.

### First Experiment (Day 1)
Generate 150 Hindi/Kannada synthetic response pairs via few-shot prompting. Pass these through quality/safety screening by the native reviewer. If acceptable, we proceed to create the training set and the Day 5 SFT pilot.
