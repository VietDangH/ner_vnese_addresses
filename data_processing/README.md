# data_processing — dataset cleaning pipeline

Reproducible, per-dataset preprocessing for the four address sources.

Each dataset runs the same staged pipeline:

```
load raw
  → dataset-specific structural cleaning      (per script)
  → shared normalization                      (NFC, de-emoji, lower, whitespace, de-dup)
  → EDA (token stats + city tails + histogram)
  → VnCoreNLP word segmentation
  → address-specific regex number fix-ups     (894H→894 H, 7/4→7 / 4, 457 A→457A)
  → data<N>_final.csv
```

It stops **before** the label-time *"augmentation as features"* step (keyword
abbreviation `quận`→`q.`, prefix drop, diacritic removal), which is applied later
to the train/test splits by `vn_ner.augment` / `bami_ner.augment`.

## Files

| File | Role |
|------|------|
| `common.py` | Shared cleaning + segmentation + EDA + gazetteer sampler (imported by all) |
| `preprocess_house_rent.py` | **Dataset 1** — Vietnam House Rent (+ geographic rebalancing) |
| `preprocess_vietnamese_address_parser.py` | **Dataset 2** — vietnamese-address-parser (structural repair) |
| `preprocess_xe_du.py` | **Dataset 3** — Xe dù delivery set (stack sender+receiver) |
| `crawl_masothue.py` | **Dataset 4** — masothue.com (crawler + shared pass) |

## Per-dataset cleaning (what differs)

- **House Rent** — well-formatted but Đà Nẵng-heavy, so we **rebalance
  geographically**: truncate each address at its ward and append a random
  ward/district/city cluster from the gazetteer, biased toward rarer provinces
  (`freq(city) ** -(1+bias)`, big-4 provinces excluded). Keeps the house/street
  prefix. Toggle with `--no-rebalance`, tune with `--rare-bias`.
- **vietnamese-address-parser** — noisy: split on newlines/pipes, strip trailing
  country tokens, collapse semantic duplicates via an abbreviation dictionary
  (`TP. Hồ Chí Minh`≡`hcm`, `Bà Rịa-Vũng Tàu`≡`brvt`), drop `#ERROR!` and
  single-token rows.
- **Xe dù** — stack each record's `senderAddress` + `receiverAddress` into one
  address-per-row stream, then shared pass. Trailing `vietnam` / `ho chi minh
  city` tokens are kept (report = shared pass only); use `--strip-country` to drop
  the last comma-segment.
- **masothue** — already standardised; drop the trailing `, Việt Nam` segment,
  then shared pass. `--crawl` re-scrapes the raw file.

## Usage

Run from the repo root (`_organized/`) so imports resolve:

```bash
python -m data_processing.preprocess_house_rent
python -m data_processing.preprocess_vietnamese_address_parser
python -m data_processing.preprocess_xe_du
python -m data_processing.crawl_masothue                 # preprocess existing raw file
python -m data_processing.crawl_masothue --crawl         # re-scrape masothue.com first
```

Common flags: `--no-segment` (skip word segmentation), `--no-eda`, `--input`,
`--output`. See each script's `--help`.

- **Inputs:** `../data/raw_data/` (the four raw CSVs) + `../data/gazetteer.csv`
- **Outputs:** `../data/processed/data{1,2,3}_final.csv`, `masothue_final.csv`
- **EDA:** `../data/processed/eda/` (token histograms + city-tail CSVs)