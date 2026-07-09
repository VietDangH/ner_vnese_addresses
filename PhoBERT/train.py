# -*- coding: utf-8 -*-
"""Training entry point.

Choose the architecture on the command line — the head selects "with or without
CRF", the top encoder optionally stacks a BiLSTM/CNN on the PhoBERT features:

    # PhoBERT + CRF (default)
    python -m vn_ner.train --train_data train.conll --val_data val.conll

    # PhoBERT only (no CRF)
    python -m vn_ner.train --train_data train.conll --val_data val.conll --head softmax

    # PhoBERT + BiLSTM + CRF
    python -m vn_ner.train --train_data train.conll --val_data val.conll \
        --top_encoder bilstm --lstm_hidden 256 --head crf

Data splitting and noise augmentation are separate steps — see ``vn_ner.split``
and ``vn_ner.augment``.
"""

import argparse
import sys

from . import config as C
from .config import TrainConfig
from .models import ENCODER_REGISTRY, HEAD_REGISTRY
from .trainer import train

# UTF-8 console (a cp1252 Windows terminal would crash on Vietnamese output).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a PhoBERT NER model (choose head + optional top layer)")
    # data
    p.add_argument("--data", default=C.DEFAULT_DATA,
                   help="single file split internally into train/val (quick mode)")
    p.add_argument("--train_data", default=None, help="pre-split train file")
    p.add_argument("--val_data", default=None, help="pre-split val file")
    p.add_argument("--val_ratio", type=float, default=C.DEFAULT_VAL_RATIO,
                   help="val fraction when using --data (quick mode)")
    # architecture
    p.add_argument("--head", default=C.DEFAULT_HEAD, choices=sorted(HEAD_REGISTRY),
                   help="decoder head: 'crf' (with CRF) or 'softmax' (without)")
    p.add_argument("--top_encoder", default=C.DEFAULT_TOP_ENCODER,
                   choices=sorted(ENCODER_REGISTRY),
                   help="optional layer on top of PhoBERT before the head")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="freeze PhoBERT; train only the top/head layers")
    # top-encoder hyperparams (used only by the matching --top_encoder)
    p.add_argument("--lstm_hidden", type=int, default=256)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--cnn_channels", type=int, default=256)
    p.add_argument("--cnn_kernel", type=int, default=3)
    p.add_argument("--cnn_layers", type=int, default=1)
    p.add_argument("--top_dropout", type=float, default=0.0)
    # model / optimisation
    p.add_argument("--model_name", default=C.DEFAULT_MODEL_NAME)
    p.add_argument("--output_dir", default=C.DEFAULT_OUTPUT_DIR)
    p.add_argument("--max_length", type=int, default=C.DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=C.DEFAULT_BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=C.DEFAULT_EPOCHS)
    p.add_argument("--lr_bert", type=float, default=C.DEFAULT_LR_BERT)
    p.add_argument("--lr_head", type=float, default=C.DEFAULT_LR_HEAD)
    p.add_argument("--weight_decay", type=float, default=C.DEFAULT_WEIGHT_DECAY)
    p.add_argument("--warmup_ratio", type=float, default=C.DEFAULT_WARMUP_RATIO)
    p.add_argument("--max_grad_norm", type=float, default=C.DEFAULT_MAX_GRAD_NORM)
    p.add_argument("--dropout", type=float, default=C.DEFAULT_DROPOUT)
    p.add_argument("--num_workers", type=int, default=C.DEFAULT_NUM_WORKERS)
    p.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    p.add_argument("--patience", type=int, default=C.DEFAULT_PATIENCE)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--no_fp16", action="store_true",
                   help="disable mixed precision (use on CPU)")
    return p.parse_args()


def _encoder_kwargs(a):
    if a.top_encoder == "bilstm":
        return {"hidden": a.lstm_hidden, "num_layers": a.lstm_layers,
                "dropout": a.top_dropout}
    if a.top_encoder == "cnn":
        return {"channels": a.cnn_channels, "kernel_size": a.cnn_kernel,
                "num_layers": a.cnn_layers, "dropout": a.top_dropout}
    return None


def main():
    a = parse_args()
    cfg = TrainConfig(
        data=a.data, train_data=a.train_data, val_data=a.val_data,
        val_ratio=a.val_ratio, model_name=a.model_name, output_dir=a.output_dir,
        head=a.head, top_encoder=a.top_encoder, encoder_kwargs=_encoder_kwargs(a),
        freeze_encoder=a.freeze_encoder,
        max_length=a.max_length, batch_size=a.batch_size, epochs=a.epochs,
        lr_bert=a.lr_bert, lr_head=a.lr_head, weight_decay=a.weight_decay,
        warmup_ratio=a.warmup_ratio, max_grad_norm=a.max_grad_norm,
        dropout=a.dropout, num_workers=a.num_workers, seed=a.seed,
        patience=a.patience, grad_accum=a.grad_accum, fp16=not a.no_fp16,
    )
    train(cfg)


if __name__ == "__main__":
    main()
