# Part B: Capacity Reconciliation

## B1. KV Cache Capacity

**a) KV-cache bytes per token, exactly:**
The model uses Grouped-Query Attention (GQA) with FP16 precision. For each token, we store the Key (K) and Value (V) tensors across all layers.
* 2 tensors (K and V)
* 28 layers
* 8 KV heads
* 128 head dimension
* 2 bytes per element (FP16)

`Bytes per token = 2 * 28 * 8 * 128 * 2 = 114,688 bytes`

**b) Maximum concurrent 4096-token sequences:**
We explicitly use decimal gigabytes (1 GB = 10^9 bytes) consistently, following the model specification's 24 GB footprint.
* **GPU capacity:** 24 GB = `24,000,000,000` bytes
* **gpu_memory_utilization:** 0.92
* **Usable GPU Memory:** `24,000,000,000 * 0.92` = `22,080,000,000` bytes
* **Model weight memory:** 4.2B params * 2 bytes = `8,400,000,000` bytes
* **Runtime overhead:** 1.6 GB = `1,600,000,000` bytes
* **Remaining KV memory:** `22,080,000,000 - 8,400,000,000 - 1,600,000,000` = `12,080,000,000` bytes
* **KV bytes/4096-token sequence:** `114,688 * 4096` = `469,762,048` bytes
* **Theoretical sequence capacity:** `12,080,000,000 / 469,762,048` = **~25.7 sequences**

*Observed-utilization extrapolation/sanity check:* 
In `bench_log.csv`, Row 12 (batch 24) reports a `kv_cache_util` of 0.93. `24 / 0.93` yields an extrapolated capacity of **~25.8 sequences**. This provides a useful sanity check: the theoretical 25.7 closely aligns with the observed-utilization extrapolation of 25.8, strongly supporting that the hardware cannot hold 32 sequences.

## B2. Long-context throughput anomaly

**The Anomaly:** In the long-context sweep (prompt=3584), throughput strictly increases up to batch 24. But at batch 32 and 48, throughput plummets, breaking the naive expectation that throughput scales with batch size.
* Row 12 (batch 24): 1607.4 tok/s (0 preemptions)
* Row 13 (batch 32): 1384.0 tok/s (7 preemptions)
* Row 14 (batch 48): 1298.5 tok/s (23 preemptions)

**Mechanism:** The log supports KV-cache pressure and scheduler preemption as the mechanism. As shown in B1, the GPU can only hold ~26 sequences of this length. At batch 32, `kv_cache_util` hits a ceiling (0.97) and `preempted_seqs` jumps from 0 to 7. The serving engine cannot fit all sequences in the GPU KV cache, forcing it to preempt sequences. 

The throughput degradation is substantial:
* **Batch 24 → 32:** throughput falls **13.9%** (1607.4 → 1384.0), while preempted sequences rise 0 → 7.
* **Batch 24 → 48:** throughput falls **19.2%** (1607.4 → 1298.5), while preempted sequences rise 0 → 23.

Note: We cannot claim direct CPU offloading bandwidth caused the slowdown without a swap-specific measurement, but Preemption is clearly supported by the log; the exact amount of recomputation or swapping would require the serving-stack metric proposed in B4.

**Proposed Change:** Cap `max_num_seqs` at 24 for the 3584-token workload.
*Prediction:* This should eliminate the observed preemptions at batch 32/48 by keeping concurrency within the observed non-preempting regime. We expect throughput to return toward the batch-24 reference of ~1607 reported tok/s, although this must be validated with a rerun.

## B3. Re-evaluating REPORT_v0

**The Misreading:** The previous intern read the `reported_tok_s` column and concluded that longer prompts yield higher throughput. However, `reported_tok_s` measures total tokens (prompt + generated tokens). Because the prefill phase processes massive prompt blocks in parallel at very high speeds, large prompts artificially inflate `reported_tok_s`. 

**Deriving Honest Goodput (Batch 24, Long Prompt):**
"Goodput" only counts the actual generated (output) tokens delivered to the user. We derive it in two independent ways for Row 12:
1. **Total generated tokens / Wall clock:** `(num_requests * gen_len) / wall_clock_s`
   = `(24 * 512) / 61.16` = **200.9 tok/s**
2. **From reported total throughput:** `reported_tok_s * (gen_len / (prompt_len + gen_len))`
   = `1607.4 * (512 / 4096)` = `1607.4 / 8` = **200.9 tok/s**

Both calculations agree. For batch 24, there are 24 × (3584 + 512) = 98,304 total tokens. Dividing by 61.16 seconds gives ~1607 tok/s. But only 24 × 512 = 12,288 tokens are generated, giving ~201 generated tok/s.

**What the report should have said:**
"Longer prompts yield substantially lower generation throughput (201 tok/s at batch 24 vs 294 tok/s at batch 16 for short prompts). Furthermore, due to hard KV cache limits, attempting a batch size of 48 for long prompts causes scheduler preemption, dropping throughput further. The batch-48 long-prompt configuration delivers 1298.5 reported tok/s in this benchmark and therefore does not support the report's projected ~3200 tok/s claim."

## B4. Confirming the Mechanism

To identify the exact serving mechanism behind the preemption-related slowdown, I would pull the serving stack's swap/preemption metric.
* **Metric:** Number of currently swapped/preempted requests exposed by the serving stack (e.g., `vllm:num_requests_swapped` for legacy vLLM, or the corresponding request-state running/waiting/swapped metrics for current vLLM V1).
* **Expected:** Approximately zero at batch sizes 1-24, and materially positive at batches 32 and 48. If the swapped value remains 0 while `preempted_seqs` > 0, it indicates the engine recomputed the sequences from scratch rather than offloading them to CPU, identifying the exact sub-mechanism of the performance drop.
