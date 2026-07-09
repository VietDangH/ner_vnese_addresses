# -*- coding: utf-8 -*-
"""Training pipeline: data -> model -> fit -> save best checkpoint.

Handles GPU/CPU transparently, mixed precision (fp16) on CUDA, gradient
accumulation, a warm-up + linear-decay schedule, discriminative learning rates
(lower for the pretrained encoder, higher for the fresh top/head layers), and
early stopping on validation entity-F1. The best checkpoint (by val F1) is what
gets saved.

Scope note: training does **not** score a test set — that is the job of
``vn_ner.evaluate`` (run it on the held-out test file afterwards). Data splitting
and noise augmentation are also separate steps (``vn_ner.split`` /
``vn_ner.augment``); the trainer just consumes the TRAIN and VAL files it is given.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import TrainConfig
from .data import (Collator, NERDataset, build_label_list, read_conll,
                   split_sentences)
from .metrics import compute_metrics
from .models import build_model


def _autocast(use_amp: bool):
    """Mixed-precision context, using the modern torch.amp API when available."""
    try:
        return torch.amp.autocast("cuda", enabled=use_amp)
    except (AttributeError, TypeError):              # older torch
        return torch.cuda.amp.autocast(enabled=use_amp)


def _grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr_bert, lr_head, weight_decay):
    """Discriminative LRs: PhoBERT encoder weights vs. the fresh top/head layers,
    with the usual no-weight-decay treatment for biases and LayerNorm."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    bert_decay, bert_no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            (bert_no_decay if any(nd in name for nd in no_decay) else bert_decay).append(p)
        else:                                   # top encoder + head
            head.append(p)
    groups = [
        {"params": bert_decay, "lr": lr_bert, "weight_decay": weight_decay},
        {"params": bert_no_decay, "lr": lr_bert, "weight_decay": 0.0},
        {"params": head, "lr": lr_head, "weight_decay": 0.0},
    ]
    # drop empty groups (e.g. when the encoder is frozen)
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups)


@torch.no_grad()
def run_inference(model, loader, id2label, device, use_amp=False):
    """Decode a whole loader -> ``(y_true, y_pred, avg_loss)``."""
    model.eval()
    y_true, y_pred = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast(use_amp):
            out = model(**batch, decode=True)
        total_loss += out["loss"].item()
        n_batches += 1
        preds = out["preds"]
        gold = batch["labels"]
        lengths = batch["word_mask"].sum(dim=1).tolist()
        for i, n in enumerate(lengths):
            n = int(n)
            y_pred.append([id2label[t] for t in preds[i]])
            y_true.append([id2label[t] for t in gold[i, :n].tolist()])
    return y_true, y_pred, total_loss / max(n_batches, 1)


def _make_loader(data, collate, batch_size, shuffle, num_workers, pin):
    return DataLoader(NERDataset(data), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=num_workers, pin_memory=pin)


def _dedup(sentences):
    seen, deduped = set(), []
    for s in sentences:
        key = (tuple(s[0]), tuple(s[1]))
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def train(cfg: TrainConfig):
    """Run training described by ``cfg``. Returns a summary dict with best val F1."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.fp16 and device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | mixed precision: {use_amp}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- data: TRAIN + VAL only ----------------------------------------- #
    if cfg.train_data and cfg.val_data:
        train_data = read_conll(cfg.train_data)
        val_data = read_conll(cfg.val_data)
        print(f"Pre-split files -> train {len(train_data)} ({cfg.train_data}) | "
              f"val {len(val_data)} ({cfg.val_data})")
    else:
        sentences = _dedup(read_conll(cfg.data))
        print(f"Loaded {len(sentences)} unique sentences from {cfg.data}")
        # single-file convenience: split into train/val only (no test slice)
        train_data, val_data, _ = split_sentences(
            sentences, cfg.val_ratio, 0.0, cfg.seed)
        print(f"Split -> train {len(train_data)} | val {len(val_data)}")

    label_list = build_label_list(train_data + val_data)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    cfg.label2id = label2id
    cfg.id2label = {str(k): v for k, v in id2label.items()}
    print(f"{len(label_list)} labels: {label_list}")
    print(f"Architecture -> head={cfg.head} | top_encoder={cfg.top_encoder}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    collate = Collator(tokenizer, label2id, cfg.max_length)
    pin = device.type == "cuda"
    train_loader = _make_loader(train_data, collate, cfg.batch_size, True,
                                cfg.num_workers, pin)
    val_loader = _make_loader(val_data, collate, cfg.batch_size, False,
                              cfg.num_workers, pin)

    # ---- model / optim -------------------------------------------------- #
    model = build_model(cfg.model_name, len(label_list), head=cfg.head,
                        top_encoder=cfg.top_encoder, dropout=cfg.dropout,
                        encoder_kwargs=cfg.encoder_kwargs,
                        head_kwargs=cfg.head_kwargs).to(device)
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen -> training top/head layers only")

    optimizer = build_optimizer(model, cfg.lr_bert, cfg.lr_head, cfg.weight_decay)
    steps_per_epoch = max(len(train_loader) // cfg.grad_accum, 1)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    scaler = _grad_scaler(use_amp)

    # ---- output dir / persisted config (arch + labels = reloadable) ----- #
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": cfg.model_name,
            "max_length": cfg.max_length,
            "head": cfg.head,
            "top_encoder": cfg.top_encoder,
            "encoder_kwargs": cfg.encoder_kwargs,
            "head_kwargs": cfg.head_kwargs,
            "label2id": label2id,
            "id2label": cfg.id2label,
        }, f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(out_dir / "tokenizer")

    # ---- training loop -------------------------------------------------- #
    try:
        from tqdm.auto import tqdm
    except Exception:                              # pragma: no cover
        def tqdm(x, **k):
            return x

    best_f1, best_epoch, epochs_no_improve = -1.0, 0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader),
                    desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for step, batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with _autocast(use_amp):
                loss = model(**batch)["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()
            if step % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            running += loss.item() * cfg.grad_accum
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{running / step:.4f}")

        y_true, y_pred, val_loss = run_inference(model, val_loader, id2label,
                                                 device, use_amp)
        m = compute_metrics(y_true, y_pred)
        vf1 = m["entity"]["f1"]
        print(f"Epoch {epoch:>2} | {time.time() - t0:>5.0f}s | "
              f"train_loss {running / max(len(train_loader),1):.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val P {m['entity']['precision']:.4f} "
              f"R {m['entity']['recall']:.4f} F1 {vf1:.4f}")
        history.append({"epoch": epoch,
                        "train_loss": running / max(len(train_loader), 1),
                        "val_loss": val_loss, "val_f1": vf1})

        if vf1 > best_f1:
            best_f1, best_epoch, epochs_no_improve = vf1, epoch, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
            print(f"   ^ new best (val F1 {best_f1:.4f}) saved to "
                  f"{out_dir / 'model.pt'}")
        else:
            epochs_no_improve += 1
            if cfg.patience and epochs_no_improve >= cfg.patience:
                print(f"   early stopping (no val-F1 gain for {cfg.patience} "
                      f"epochs; best was epoch {best_epoch})")
                break

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nBest val F1 {best_f1:.4f} at epoch {best_epoch}. "
          f"Artifacts in {out_dir.resolve()}")
    print(f"Next: python -m vn_ner.evaluate --model_dir {cfg.output_dir} "
          f"--data <your_test.conll>")
    return {"best_val_f1": best_f1, "best_epoch": best_epoch}
