# MarkLLM Watermark Survey

**Project:** Benchmark-style survey evaluating scale effects in LLM watermarking (KGW, Unigram, EXP) across model sizes (OPT-125M/1.3B/2.7B) and datasets (C4, WMT16).

---

## Quick Structure

```
watermark_survey/
├── generated_texts/        # 36 JSONL files: {algo}_{model}_{dataset}_{wm|uwm}.jsonl (3,600 samples)
├── attack_outputs/         # 36 JSONL: word_deletion, synonym (BERT), paraphrase (GPT-3.5)
├── batch_api/              # paraphrase_batch_v2.jsonl (input), paraphrase_batch_v2_results.jsonl (output)
└── results/
    ├── scores/             # 108 JSONL: detect_{combo}_{condition}_{wm|uwm}.jsonl, ppl_{combo}_{wm|uwm}.jsonl
    ├── tables/             # CSV/MD/TeX: detectability, robustness, quality, + caveats.json
    └── JSON summaries      # robustness.json, quality_ppl.json, quality_log_diversity*, metrics_cross_validation.json
```

**Key combos:** 18 for generation/detection; 12 for robustness (OPT-125M excluded—baseline detection too low).

---

## Critical Design Decisions

1. **Generation precision:** All OPT in fp16 in demo; shipped `generated_texts/` are canonical (mixed fp32/fp16 from original run). Re-running Phase 1 yields statistically equivalent but not byte-identical text.

2. **Detection metric:** Custom `compute_metrics` (not MarkLLM's `DynamicThresholdSuccessRateCalculator`) for explicit EXP polarity handling and diagnostics. Cross-validated against MarkLLM: TPR=0.99 exact match on KGW+OPT-1.3B+C4.

3. **Score polarity:** `POLARITY = {"KGW": True, "Unigram": True, "EXP": False}` — KGW/Unigram: higher z-score = watermarked; EXP: lower p-value = watermarked.

4. **OPT-125M:** In detectability/quality tables (TPR≈0.10 shows scale-dependent floor). Excluded from robustness (attacking undetectable signal meaningless).

5. **WMT16 perplexity:** Omitted from Table 3 (GPT-2 is English-trained; high PPL on German = language mismatch, not watermark cost). Raw values in `quality_ppl.json` + caveat in `caveats.json`.

6. **Unigram negative ΔPPL on C4:** Green list overlaps GPT-2 high-probability tokens; reported with discussion flag.

7. **Robustness negative class:** Uses clean `uwm` scores (not attacked-uwm) for TPR/F1; 10% FPR threshold calibrated against natural text.

8. **Log-diversity:** Two aggregations—corpus-level (pools n-grams across 100 texts) and per-text (avg per-sample). OPT-125M WMT16 values inflated by partial degeneracy.

---

## Reproduction Paths

**Path A (replay, ~30 min on T4):** Use shipped `generated_texts/` + `batch_api/paraphrase_batch_v2_results.jsonl`. Set `USE_DRIVE = False`, run Phase 2+ in `watermark_survey_demo.ipynb`. Numbers match report exactly.

**Path B (from scratch, ~3h):** Regenerate Phase 1 text, Phases 2–4. Each phase resumable by line count. Phase 3c consumes batch results (regen optional via OpenAI Batch API, ~$2 cost).

---

## Environment

Colab T4 (16 GB VRAM). Fresh MarkLLM clone from `https://github.com/THU-BPM/MarkLLM` (no commit SHA pinned). Deps: `scipy>=1.14,<1.16`, `transformers==4.41.2`. GPT-2 (small) and `bert-large-uncased` auto-download on first use.

---

## File Naming

```
{algorithm}_{safe_model}_{dataset}_{wm|uwm}.jsonl
safe_model = HuggingFace path with / → _ (e.g. facebook_opt-1.3b)
combo = algorithm_safe_model_dataset
condition ∈ {clean, word_deletion, synonym, paraphrase}
```

---

## What Agent Should Know

- **For code/notebook edits:** Refer to `watermark_survey_demo.ipynb` phases and the four directories above.
- **For reproducibility Q's:** Cite methodological notes on fp16, polarity, metrics cross-validation, OPT-125M scope.
- **For table interpretation:** Check `caveats.json` (WMT16 PPL omission, Unigram ΔPPL anomaly).
- **For data ownership:** All intermediates shipped except MarkLLM framework itself (cloned fresh).
