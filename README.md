# Vietnamese Address NER

Named-entity recognition for Vietnamese addresses. Given a free-form address
string, the models tag each part with its role — house number, street, ward,
district, city, and more:

```
123 đường lê lợi, phường bến nghé, quận 1, tp hcm
└HOUSE_NO┘ └─STREET─┘  └───WARD────┘ └DISTRICT┘ └CITY┘
```

**Entity types:** `HOUSE_NO`, `ALLEY`, `STREET`, `AREA`, `BUILDING`, `WARD`,
`DISTRICT`, `CITY` (BIO-tagged, CoNLL format).

## What's inside

The repo covers the full pipeline — from raw data to a served API — with several
interchangeable modelling approaches so they can be trained and benchmarked
through an identical metrics path.

| Component | What it does |
|-----------|--------------|
| [`data_processing/`](data_processing/README.md) | Cleans + normalizes + word-segments four raw address datasets into training data |
| [`PhoBERT/`](PhoBERT/README.md) | PhoBERT model on **word-segmented** text; pluggable top layer (BiLSTM/CNN) and head (softmax/CRF) |
| [`BamiBERT/`](BamiBERT/README.md) | Same architecture on **raw syllables** (no word segmenter needed) |
| [`rule_based model/`](rule_based%20model/README.md) | Deterministic gazetteer + trigger-word tagger; a no-training baseline |
| [`fastapi_serving/`](fastapi_serving/README.md) | REST API (FastAPI) for single / batch / file prediction |
| [`triton_serving/`](triton_serving/README.md) | Triton Inference Server serving the same model, identical outputs |
| [`VnCoreNLP-master/`](VnCoreNLP-master/) | Vendored VnCoreNLP word segmenter used by PhoBERT |
| `data/` | `Cleaned_Data.conll` (6,820 sentences), gazetteer, raw sources, and seeded train/val/test splits |

## Quick start

Each model package is a self-contained CLI following the same steps:
**split → augment (optional) → train → evaluate → predict / benchmark**.

```bash
# Train a PhoBERT + CRF model on a seeded split
python -m PhoBERT.split    --data data/Cleaned_Data.conll --out_dir data/splits/data_split_seed_42
python -m PhoBERT.train    --train_data data/splits/data_split_seed_42/train.conll \
                           --val_data   data/splits/data_split_seed_42/val.conll --head crf
python -m PhoBERT.evaluate --model_dir ner_model --data data/splits/data_split_seed_42/test.conll
python -m PhoBERT.predict  --model_dir ner_model --text "123 đường lê lợi, quận 1, tp hcm"
```

Serve a trained model:

```bash
uvicorn fastapi_serving.app:app --reload     # → http://localhost:8000/docs
```

See each subfolder's README for the full workflow, architecture flags, and
serving details.