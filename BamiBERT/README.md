# Vietnamese Address NER on BamiBERT

```
BamiBERT encoder  ─►  [ optional top layer ]  ─►  [ head ]  ─►  loss / tags
   (raw text)           none / bilstm / cnn        softmax / crf
```

## Input: de-segment first

The dataset is word-segmented for PhoBERT (compounds joined with `_`, e.g.
`thành_phố`). BamiBERT needs raw syllables, so run the standalone de-segmenter
(outside the package) once:

```bash
python desegment.py --data Cleaned_Data_new.conll --out Cleaned_Data_new_raw.conll
```

It splits every `_`-joined token into syllables and expands the BIO labels
(`thành_phố B-DISTRICT` → `thành B-DISTRICT` + `phố I-DISTRICT`), then verifies
that **every entity span (type + surface text) is unchanged** before writing.

## Workflow

```bash
# 0) de-segment (segmented -> raw); only needed once
python desegment.py --data Cleaned_Data_new.conll --out Cleaned_Data_new_raw.conll

# 1) split raw data -> train/val/test (all clean)
python -m bami_ner.split --data Cleaned_Data_new_raw.conll --out_dir data_split

# 2) (optional) augment ANY file — usually train
python -m bami_ner.augment --data data_split/train.conll --out data_split/train_aug.conll

# 3) train — pick the head (with/without CRF) and an optional top layer
python -m bami_ner.train --train_data data_split/train.conll --val_data data_split/val.conll --head crf
python -m bami_ner.train --train_data data_split/train.conll --val_data data_split/val.conll --head softmax
python -m bami_ner.train --train_data data_split/train.conll --val_data data_split/val.conll \
    --top_encoder bilstm --lstm_hidden 256 --head crf

# 4) evaluate on the held-out test file (architecture auto-detected from config.json)
python -m bami_ner.evaluate --model_dir bami_ner_model --data data_split/test.conll

# 5) predict / benchmark  (raw text — no word segmentation; benchmark times the
#    full pipeline: raw text -> whitespace/punct tokenize -> model -> tags)
python -m bami_ner.predict   --model_dir bami_ner_model --text "123 đường lê lợi, quận 1, tp hcm"
python -m bami_ner.benchmark --model_dir bami_ner_model --data data_split/test.conll
```

Use a separate `--output_dir` per architecture to keep them side by side, e.g.
`--head crf --output_dir crf_model` vs `--head softmax --output_dir softmax_model`.

## Choosing the architecture

| Flag | Choices | Meaning |
|------|---------|---------|
| `--head` | `crf`, `softmax` | with / without CRF |
| `--top_encoder` | `none`, `bilstm`, `cnn` | extra layer on BamiBERT features |
| `--freeze_encoder` | (flag) | train only top/head (probe BamiBERT) |

Top-layer hyperparams: `--lstm_hidden/--lstm_layers`, `--cnn_channels/--cnn_kernel/--cnn_layers`, `--top_dropout`.

The chosen head/top_encoder (and their kwargs) are saved into `config.json`, so
`evaluate` / `predict` / `benchmark` rebuild the exact model automatically.