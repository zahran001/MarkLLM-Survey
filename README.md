# Watermark Survey — Scale Effects in LLM Watermarking

A benchmark-style survey using the [MarkLLM framework](https://github.com/THU-BPM/MarkLLM) (Pan et al., 2024) to evaluate how three representative watermarking algorithms (KGW, Unigram, EXP) behave across three model scales (OPT-125M, OPT-1.3B, OPT-2.7B) and two datasets (C4, WMT16). The central research question is whether watermarking behavior — detectability, robustness, and quality cost — changes with model scale.

**Authors:** Zahran Yahia Khan, Sofia Cobo Navas

---

## What's in this repo

```
MarkLLM/
├── README.md
├── watermark_survey_demo.ipynb       # Single-notebook reproduction of the full pipeline
└── watermark_survey/
    ├── generated_texts/              # 36 files — Phase 1 output (3,600 samples)
    │                                 #   {algorithm}_{safe_model}_{dataset}_{wm|uwm}.jsonl
    ├── attack_outputs/
    │   ├── word_deletion/            # 12 files — Phase 3a output
    │   ├── synonym/                  # 12 files — Phase 3b output
    │   └── paraphrase/               # 12 files — Phase 3c output (parsed from batch results)
    ├── batch_api/
    │   ├── paraphrase_batch_v2.jsonl         # OpenAI batch input (1,200 prompts)
    │   └── paraphrase_batch_v2_results.jsonl # Batch output, consumed by Phase 3c
    └── results/
        ├── scores/                   # 108 files — per-sample detection + perplexity
        │                             #   detect_{combo}_{condition}_{wm|uwm}.jsonl  (72)
        │                             #   ppl_{combo}_{wm|uwm}.jsonl                 (36)
        ├── tables/
        │   ├── detectability.{csv,md,tex}
        │   ├── robustness.{csv,md,tex}
        │   ├── quality.{csv,md,tex}
        │   └── caveats.json
        ├── robustness.json                    # 36-row aggregated robustness table
        ├── quality_ppl.json                   # GPT-2 perplexity, per combo
        ├── quality_log_diversity.json         # corpus-level log-div
        ├── quality_log_diversity_per_text.json
        └── metrics_cross_validation.json      # compute_metrics ↔ MarkLLM calculator
                                               # (TPR=0.99 exact match)
```

The repo ships every intermediate artifact needed to verify the report's numbers, including the OpenAI batch results. Path A below replays the pipeline against these files; Path B regenerates from scratch.

---

## Scope

|              | Values                                                       |
|--------------|--------------------------------------------------------------|
| Algorithms   | KGW, Unigram, EXP                                            |
| Models       | facebook/opt-125m, facebook/opt-1.3b, facebook/opt-2.7b      |
| Datasets     | C4, WMT16 (de→en source side as prompt)                      |
| Samples/combo | 100 watermarked + 100 unwatermarked                         |
| Attacks      | Word deletion (0.3), synonym (BERT, 0.5), GPT-3.5 paraphrase |

**18 combos** for generation and detection. **12 combos** for robustness — OPT-125M is excluded because its baseline detection rate is too low for degradation to be meaningful (this is itself one of the findings; see methodological notes).

---

## Reproducing the results

### Path A — replay from saved files

Replays detection, aggregation, and table generation from the saved intermediate files. Numbers will match the report exactly. Total runtime ~30 min on a Colab T4.

1. Clone the repo into Colab:
   ```
   !git clone <repo URL> /content/MarkLLM_survey
   ```
2. Open `watermark_survey_demo.ipynb`. In the setup cell, set:
   ```python
   USE_DRIVE = False
   BASE_PATH = "/content/MarkLLM_survey/watermark_survey"
   ```
3. Run the setup cell.
4. Run from "Phase 2 — Clean Detection" to the end. Skip Phase 1.

If you prefer running against your own Drive copy: upload `watermark_survey/` to your Drive root, leave `USE_DRIVE = True`, and the default `BASE_PATH = "/content/drive/MyDrive/watermark_survey"` will work.

### Path B — full run from scratch

Total runtime ~3h on a Colab T4 (mostly Phase 1). Numbers will be statistically equivalent to the report but not byte-identical (see "Methodological notes" on dtype below).

1. Steps 1–3 of Path A.
2. Run all phases in order. Each phase is resumable — if Colab disconnects, re-run the cell and it resumes from the last completed sample by line count.
3. Phase 3c (paraphrase) consumes `batch_api/paraphrase_batch_v2_results.jsonl`. The repo ships this file; if you want to regenerate it from your own batch run, see the next section.

---

## Regenerating the paraphrase batch results (optional)

The shipped `paraphrase_batch_v2_results.jsonl` is the output of an OpenAI Batch API job that paraphrases all 1,200 watermarked samples (12 attack-eligible combos × 100 samples) using `gpt-3.5-turbo`. Each row's `custom_id` is `{algorithm}_{safe_model}_{dataset}_{NNN}`.

To rebuild it:

```python
import glob, os, json

wm_files = sorted(
    glob.glob(os.path.join(BASE_PATH, "generated_texts", "*opt-1.3b*_wm.jsonl")) +
    glob.glob(os.path.join(BASE_PATH, "generated_texts", "*opt-2.7b*_wm.jsonl"))
)
batch_rows = []
for fpath in wm_files:
    combo_id = os.path.basename(fpath).replace("_wm.jsonl", "")
    with open(fpath) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            batch_rows.append({
                "custom_id": f"{combo_id}_{i:03d}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user",
                                  "content": f"Paraphrase the following text while preserving its meaning:\n\n{item['text']}"}],
                    "max_tokens": 300,
                },
            })
with open(os.path.join(BASE_PATH, "batch_api", "paraphrase_batch_v2.jsonl"), "w") as f:
    for row in batch_rows:
        f.write(json.dumps(row) + "\n")
```

Upload the resulting file at <https://platform.openai.com/batches>, then save the completed results to `batch_api/paraphrase_batch_v2_results.jsonl`. Total cost was under $2 with the 50% Batch API discount; turnaround was a few hours.

---

## Environment

Tested on Colab with a T4 GPU (16 GB VRAM). The setup cell installs MarkLLM's own requirements plus two pins:

```
scipy>=1.14,<1.16
transformers==4.41.2
```

Numpy is intentionally left at whatever Colab ships. The MarkLLM repository is cloned fresh into `/content/MarkLLM` from `https://github.com/THU-BPM/MarkLLM`. We pin no commit SHA — if the upstream `evaluation/tools/text_editor` module path moves, the import at the top of Phase 3a/3b will break and you will need to follow the new path.

GPT-2 (small) is loaded for perplexity scoring in Phase 4. `bert-large-uncased` is loaded for context-aware synonym substitution in Phase 3b. Both are downloaded on first use.

---

## Methodological notes

These are the design choices a precise reviewer might ask about. We document them here so they're discoverable from the repo, not buried in the report.

### Generation precision

All three OPT sizes are loaded in `torch.float16`. Our original Phase 1 was a mix — OPT-125M and OPT-1.3B were generated in fp32, OPT-2.7B in fp16, because the project evolved across notebooks before being merged. The numbers in the report come from those original mixed-precision generations, archived in `generated_texts/`. The merged demo unifies on fp16, so re-running Phase 1 from scratch produces statistically equivalent but not byte-identical text. Detection metrics aggregated over 100 samples are not measurably affected; sample-by-sample text differs.

The shipped `generated_texts/` are the canonical inputs for the reported numbers.

### Detection-metric implementation

`compute_metrics` (TPR/F1 at 10% FPR and at best-F1 threshold) is a custom function rather than MarkLLM's `DynamicThresholdSuccessRateCalculator`. Two reasons: explicit handling of EXP's inverted score polarity (lower p-value = watermarked, encoded in the `POLARITY` lookup), and auxiliary fields useful for diagnostics. Cross-validated against MarkLLM's calculator on the KGW + OPT-1.3B + C4 control combo: TPR=0.99 exact match, F1@10%FPR=0.9519230769230769 (16-decimal match), F1@best=0.99 exact match. The validation record is at `results/metrics_cross_validation.json`.

### Score polarity

```python
POLARITY = {"KGW": True, "Unigram": True, "EXP": False}  # True = higher means watermarked
```

KGW and Unigram return z-scores (higher = stronger watermark signal). EXP returns a p-value-like correlation score where lower means more watermarked. The threshold-sweep logic flips comparison direction based on this lookup.

### OPT-125M is in detectability and quality, out of robustness

OPT-125M produces a KGW TPR@10%FPR of approximately 0.10 — at the FPR level, effectively undetectable. We kept it in the detectability and quality tables because the failure itself is a finding (RQ1 evidence: detectability does depend on scale, with a sharp floor below 1B parameters). It is excluded from the robustness experiments only because attacking an undetectable signal yields meaningless degradation deltas.

### WMT16 PPL is collected but omitted from Table 3

GPT-2 (the external scorer) is English-trained and assigns high perplexity to fluent German continuations on grounds of language mismatch rather than watermark cost. Reporting WMT16 PPL would conflate two different things. The raw values are in `quality_ppl.json` for anyone who wants them; the table omits them by design. The methods caveat is in `results/tables/caveats.json`.

### Unigram's negative ΔPPL on C4

Unigram produces a *negative* ΔPPL on C4 — watermarked text has lower GPT-2 perplexity than unwatermarked. Matched-prompt eyeball comparison showed the watermarked text is not visibly higher quality. The likely explanation is that Unigram's fixed green list overlaps with GPT-2's high-probability tokens, biasing the external scorer in Unigram's favor. We report the value but flag the interaction in the discussion.

### Negative class for post-attack metrics

Robustness aggregation uses clean `uwm` scores (not attacked-uwm) as the negative class when computing post-attack TPR/F1. We do not attack unwatermarked text in this study — there is no signal there to disrupt. The 10% FPR threshold stays calibrated against natural text, which is the realistic deployment scenario.

### Log-diversity aggregation

Reported with two aggregations. **Corpus-level** pools n-grams (n=2,3,4) across all 100 texts per (combo, variant), capturing cross-text variety. **Per-text** computes log-diversity per sample then averages, capturing within-text repetition. Both directions agree (wm < uwm across embeddable scales) but they answer different questions. Absolute values differ from MarkLLM Table 4 due to differing aggregation; within-study deltas are valid.

OPT-125M log-diversity values on WMT16 are inflated by partially-degenerate output (verbatim prompt repetition followed by topic-drifting text). N-gram diversity rewards novelty without measuring coherence. Comparative claims across model scale should focus on the 1.3B → 2.7B transition.

---

## File-naming conventions

```
generated_texts/{algorithm}_{safe_model}_{dataset}_{wm|uwm}.jsonl
attack_outputs/{word_deletion|synonym|paraphrase}/{combo}.jsonl
results/scores/detect_{combo}_{condition}_{wm|uwm}.jsonl
results/scores/ppl_{combo}_{wm|uwm}.jsonl
```

Where `safe_model` is the HuggingFace name with `/` replaced by `_` (e.g. `facebook_opt-1.3b`), `combo = algorithm_safemodel_dataset`, and `condition ∈ {clean, word_deletion, synonym, paraphrase}`.

---

## Citing

If you use this work or build on it, please cite the original MarkLLM paper:

```bibtex
@inproceedings{pan2024markllm,
  title     = {MarkLLM: An Open-Source Toolkit for LLM Watermarking},
  author    = {Pan, Leyi and Liu, Aiwei and He, Zhiwei and Gao, Zitian and Zhao, Xuandong
               and Lu, Yijian and Zhou, Bingling and Liu, Shuliang and Hu, Xuming
               and Wen, Lijie and King, Irwin and Yu, Philip S.},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year      = {2024},
}
```