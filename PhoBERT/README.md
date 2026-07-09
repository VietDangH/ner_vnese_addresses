# Vietnamese Address NER on PhoBERT

```
PhoBERT encoder  ─►  [ optional top layer ]  ─►  [ head ]  ─►  loss / tags
                       none / bilstm / cnn        softmax / crf
```

## Layout

```
PhoBERT/
├── config.py        # TrainConfig + defaults (head, top_encoder, hyperparams)
├── data.py          # CoNLL IO, label vocab, split, Dataset, Collator
├── segment.py       # VN word segmentation (VnCoreNLP/underthesea/pyvi/whitespace)
├── metrics.py       # entity + token level report (the only metrics code)
├── noise.py         # noise operators (used by augment.py)
├── crf_lib.py       # pytorch-crf loader (only used by the CRF head)
├── models/
│   ├── __init__.py  # build_model(...) + load_trained(...) + registries
│   ├── base.py      # NERModel: encoder -> top -> head (stable interface)
│   ├── encoders.py  # top layers: none / bilstm / cnn   (ENCODER_REGISTRY)
│   └── heads.py     # decoders: softmax / crf           (HEAD_REGISTRY)
├── split.py         # CLI: split a clean file -> train/val/test   (no augmentation)
├── augment.py       # CLI: noise-augment ANY file                  (decoupled)
├── train.py         # CLI: train (choose --head / --top_encoder)   (no test eval)
├── evaluate.py      # CLI: the ONLY place metrics are computed
├── predict.py       # CLI: tag new addresses
└── benchmark.py     # CLI: end-to-end (raw text -> segment -> tags) timing
```

Single-purpose steps (separated on purpose):

| Step | Does only | Notably does **not** |
|------|-----------|----------------------|
| `split`    | de-dup + split into files | augment |
| `augment`  | noise any file (append/replace) | split |
| `train`    | fit + save best-by-val checkpoint | score the test set |
| `evaluate` | compute the full metrics report | train |

## Workflow

```bash
# 1) split clean data -> train/val/test (all clean)
python -m vn_ner.split --data Cleaned_Data.conll --out_dir data_split

# 2) (optional) augment ANY file — usually train; or build a hard, noisy test set
python -m vn_ner.augment --data data_split/train.conll --out data_split/train_aug.conll
python -m vn_ner.augment --data data_split/test.conll  --out data_split/test_noisy.conll --mode replace

# 3) train — pick the head (with/without CRF) and an optional top layer
python -m vn_ner.train --train_data data_split/train_aug.conll --val_data data_split/val.conll --head crf
python -m vn_ner.train --train_data data_split/train_aug.conll --val_data data_split/val.conll --head softmax
python -m vn_ner.train --train_data data_split/train_aug.conll --val_data data_split/val.conll \
    --top_encoder bilstm --lstm_hidden 256 --head crf

# 4) evaluate on the held-out test file (architecture auto-detected from config.json)
python -m vn_ner.evaluate --model_dir ner_model --data data_split/test.conll

# 5) predict / benchmark  (benchmark times the full pipeline: raw text ->
#    VnCoreNLP word segmentation -> model -> tags, so segmentation is included)
python -m vn_ner.predict   --model_dir ner_model --text "123 đường lê lợi, quận 1, tp hcm"
python -m vn_ner.benchmark --model_dir ner_model --data data_split/test.conll
python -m vn_ner.benchmark --compare crf_model/benchmark.json softmax_model/benchmark.json
```

Use a separate `--output_dir` per architecture to keep them side by side, e.g.
`--head crf --output_dir crf_model` vs `--head softmax --output_dir softmax_model`.

## Choosing the architecture

| Flag | Choices | Meaning |
|------|---------|---------|
| `--head` | `crf`, `softmax` | with / without CRF |
| `--top_encoder` | `none`, `bilstm`, `cnn` | extra layer on PhoBERT features |
| `--freeze_encoder` | (flag) | train only top/head (probe PhoBERT; widens the CRF gap) |

Top-layer hyperparams: `--lstm_hidden/--lstm_layers`, `--cnn_channels/--cnn_kernel/--cnn_layers`, `--top_dropout`.

The chosen head/top_encoder (and their kwargs) are saved into `config.json`, so
`evaluate` / `predict` / `benchmark` rebuild the exact model automatically.
