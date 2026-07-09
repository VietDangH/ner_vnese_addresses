from dataclasses import dataclass, asdict, field
from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# Defaults (also used as argparse defaults in the entry-point scripts)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_NAME = "Qualcomm-AI-Research/BamiBERT"
# Raw (de-segmented) CoNLL — produce it from a segmented file with desegment.py.
DEFAULT_DATA = "Cleaned_Data_new_raw.conll"
DEFAULT_OUTPUT_DIR = "bami_ner_model"

# Architecture choices (see bami_ner.models). Pick the head to train "with or
# without CRF"; pick a top encoder to stack BiLSTM/CNN on the BamiBERT features.
DEFAULT_HEAD = "crf"            # "crf" | "softmax" | ... (HEAD_REGISTRY)
DEFAULT_TOP_ENCODER = "none"    # "none" | "bilstm" | "cnn" | ... (ENCODER_REGISTRY)

# BamiBERT supports up to 2048 tokens; addresses are short so 256 is plenty and
# keeps runs comparable with the PhoBERT (vn_ner) pipeline.
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR_BERT = 3e-5
DEFAULT_LR_HEAD = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_DROPOUT = 0.1
DEFAULT_VAL_RATIO = 0.1
DEFAULT_NUM_WORKERS = 2
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 4


@dataclass
class TrainConfig:
    """Everything the training pipeline needs.

    Training reads a TRAIN and a VAL file (data splitting and noise augmentation
    are separate steps — see ``bami_ner.split`` and ``bami_ner.augment``). For
    quick experiments, leave ``train_data``/``val_data`` unset and pass a single
    ``data`` file: it is de-duplicated and split into train/val by ``val_ratio``.
    """

    data: str = DEFAULT_DATA
    # Pre-split files (preferred). If both are set, the internal split is skipped.
    train_data: Optional[str] = None
    val_data: Optional[str] = None

    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = DEFAULT_OUTPUT_DIR

    # ---- architecture ---- #
    head: str = DEFAULT_HEAD
    top_encoder: str = DEFAULT_TOP_ENCODER
    encoder_kwargs: Optional[Dict] = None     # e.g. {"hidden": 256, "num_layers": 1}
    head_kwargs: Optional[Dict] = None

    # ---- optimisation ---- #
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    lr_bert: float = DEFAULT_LR_BERT
    lr_head: float = DEFAULT_LR_HEAD
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    dropout: float = DEFAULT_DROPOUT
    freeze_encoder: bool = False              # train head/top only (probe BamiBERT)

    val_ratio: float = DEFAULT_VAL_RATIO
    num_workers: int = DEFAULT_NUM_WORKERS
    seed: int = DEFAULT_SEED
    patience: int = DEFAULT_PATIENCE

    fp16: bool = True           # mixed precision (auto-disabled on CPU)
    grad_accum: int = 1         # gradient-accumulation steps

    # filled in at runtime once labels are known; persisted to config.json
    label2id: Optional[Dict[str, int]] = field(default=None)
    id2label: Optional[Dict[str, str]] = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)
