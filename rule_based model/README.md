# — Rule-based Vietnamese Address NER 

It runs on the **raw, un-segmented** CoNLL data (one Vietnamese *syllable* per
token), so it has **no dependency on a word
segmenter**: multi-syllable triggers like `thành phố` / `thị trấn` / `khu phố`
are matched as phrases. It reports metrics through an **identical** code path, so
its `eval_report.txt` / `eval_metrics.json` drop straight into the same
comparison table as the neural models.

## How the rules work

Four families of rules, applied in priority order:

| # | Rule | Example |
|---|------|---------|
| 1 | **Field structure** — commas split an address into fields, ordered *specific → general* | `…, xã X, huyện Y, tỉnh Z` |
| 2 | **Leading-numeric grammar** — parse the `<alley>/<house-no>` block up front | `204 / 9` → `ALLEY / HOUSE_NO`; `k 13` → `ALLEY` |
| 3 | **Trigger words** (phrase-aware) — a closed marker set opens each field; the name after it is absorbed | `đường`→STREET, `phường`/`xã`/`thị trấn`→WARD, `quận`/`huyện`/`thị xã`→DISTRICT, `tỉnh`→CITY, `ngõ`/`hẻm`→ALLEY, `tổ`/`thôn`/`khu phố`→AREA, `chung cư`→BUILDING, `số`→HOUSE_NO |
| 4 | **Gazetteer + hierarchy** — look up trigger-less names; resolve the shared `thành phố`/`tp` prefix (CITY vs DISTRICT) by name + position | `thành phố phổ yên` → DISTRICT, `thành phố hồ chí minh` → CITY |

Lexicons live in `lexicon.py` (phrase-aware trigger matcher), the gazetteer index
in `gazetteer.py`, and the tagging state-machine in `tagger.py`.

## Layout

```
rule_ner/
├── lexicon.py     # trigger phrases + regexes + longest-phrase matcher (the knowledge)
├── gazetteer.py   # GazetteerIndex: normalise + longest-match over gazetteer.csv
├── tagger.py      # RuleBasedNER: the deterministic BIO tagger
├── data.py        # CoNLL IO + split (torch-free mirror of vn_ner.data)
├── metrics.py     # verbatim copy of vn_ner.metrics (identical report structure)
├── evaluate.py    # CLI: score a CoNLL file, write the report  ← main entry point
└── predict.py     # CLI: tag one address
```

## Usage

```bash
# Evaluate on the held-out test split of the RAW data (seed 41)
python -m rule_ner.evaluate --data Cleaned_Data_new_raw.conll --split test --seed 41

# …or score a pre-split held-out file directly
python -m rule_ner.evaluate --data raw_test.conll --split all

# Tag a single raw (un-segmented) address — one syllable per token
python -m rule_ner.predict --text "204 / 9 hải phòng , huyện cát hải , hải phòng"
```

Reports are written to `--out_dir` (default `rule_model/`) as
`eval_report.txt` and `eval_metrics.json`, same shape as the model packages.

> **Comparability:** the split is dedupe → shuffle-by-seed, identical to
> `vn_ner.split`. `Cleaned_Data_new_raw.conll` is the raw counterpart of the
> segmented `Cleaned_Data_new.conll` (same 6820 sentences, same order), so the
> same `--seed` / ratios select the corresponding held-out test set.