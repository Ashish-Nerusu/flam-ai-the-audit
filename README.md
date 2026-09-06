# Flam AI - The Audit

Evidence-based audit of multilingual tokenization and LLM serving performance for routing and capacity decisions.

## Overview

This submission audits the previous tokenizer and serving benchmark and provides corrected measurements and recommendations.

### Part A — Tokenizer Audit
- Builds a multilingual evaluation corpus covering English, Hindi, Kannada, and Telugu.
- Audits the tokenizer metric and preprocessing in `fertility.py`.
- Compares GPT-2 and XLM-R tokenization behavior.
- Evaluates multiple denominators and provides a routing/cost recommendation.

### Part B — Capacity Reconciliation
- Derives KV-cache memory requirements from the model specification.
- Reconciles theoretical capacity with benchmark results.
- Investigates long-context throughput degradation and preemption.
- Corrects the reported throughput/goodput interpretation.

### Part C — Decision Memo
- Evaluates approaches for making assistant responses more casual across six Indian languages.
- Recommends a practical SFT approach under the given compute, reviewer, and timeline constraints.

## Repository Structure

```text
├── NOTEBOOK.md
├── AI_USAGE.md
├── partA/
├── partB/
└── partC/