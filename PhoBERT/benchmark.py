# -*- coding: utf-8 -*-
"""End-to-end prediction-time benchmark — raw address string in, tags out.

Measures the *whole* inference pipeline a user actually pays for, not just the
model forward pass:

    raw text ─► word segmentation (VnCoreNLP / py_vncorenlp) ─► model ─► tags

Because PhoBERT expects word-segmented input, segmentation is a real, non-trivial
part of the latency — so it is **included** in every number here (and also broken
out on its own so you can see how much it costs). The test CoNLL tokens are
de-segmented back into a raw string first (``thành_phố`` → ``thành phố``) so the
segmenter has real work to do, exactly as at serving time.

Two views:
  * batched throughput  — segment every sentence, then run the model in batches
    at ``--batch_size``; reported end-to-end and split into segmentation vs. model.
  * single-sentence     — segment + decode one sentence at a time (latency):
    mean / median / p95 ms, plus the mean segmentation share.

    python -m vn_ner.benchmark --model_dir ner_model --data data_split/test.conll

Writes ``<model_dir>/benchmark.json`` so runs can be compared side by side
(``--compare a/benchmark.json b/benchmark.json``).

Colab: the VnCoreNLP backend needs Java + ``py_vncorenlp`` (see ``segment.py``)::

    !apt-get install -y openjdk-17-jdk-headless -qq
    !pip install py_vncorenlp

Without them the segmenter falls back to underthesea / pyvi / whitespace; the
backend actually used is printed and saved into ``benchmark.json``.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from .data import Collator, read_conll
from .models import load_trained
from .segment import get_backend, segment

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Package-specific raw-text prep — the ONLY part that differs from bami_ner.
# --------------------------------------------------------------------------- #
PREP_STAGE = "segmentation"          # what the raw-text prep step does


def prepare(text: str) -> list:
    """Raw address string -> model-ready tokens (VnCoreNLP word segmentation)."""
    return segment(text)


def to_raw_text(tokens: list) -> str:
    """Rebuild a raw address string from word-segmented CoNLL tokens.

    Undo the ``_`` compounding so the segmenter re-does the real work at bench
    time (``["thành_phố", "hà_nội"]`` -> ``"thành phố hà nội"``).
    """
    return " ".join(tokens).replace("_", " ")


# --------------------------------------------------------------------------- #
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(model, collate, device, raw_texts, warmup=2, use_amp=False,
              batch_size=32, repeats=10):
    """Time the end-to-end pipeline (prep + model). Returns a metrics dict."""
    def _autocast():
        try:
            return torch.amp.autocast("cuda", enabled=use_amp)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=use_amp)

    def _run_batch(token_lists):
        batch = collate([(toks, ["O"] * len(toks)) for toks in token_lists])
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with _autocast():
            model(input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  word_positions=batch["word_positions"],
                  word_mask=batch["word_mask"],
                  decode=True)

    model.eval()
    n_sent = len(raw_texts)

    # Warm up the full path (JVM/segmenter is already initialised via get_backend
    # in main, so this warms tokenizer caches + CUDA kernels, batched and single).
    warm = raw_texts[:max(min(warmup * batch_size, n_sent), 1)]
    warm_tokens = [prepare(t) for t in warm]
    for i in range(0, len(warm_tokens), batch_size):
        _run_batch(warm_tokens[i:i + batch_size])
    _sync(device)

    # ---- batched, end-to-end throughput ---------------------------------- #
    # Prep is done once; the model sweep is timed ``repeats`` times and the
    # MEDIAN reported. A single GPU pass on a shared / boost-clocked card swings
    # several percent run-to-run, so one sample is not reproducible — the median
    # of many is (the spread is saved as model_seconds_{min,max,std} below).
    t0 = time.perf_counter()
    prepared = [prepare(t) for t in raw_texts]
    prep_total = time.perf_counter() - t0

    model_times = []
    for _ in range(max(repeats, 1)):
        _sync(device)
        t1 = time.perf_counter()
        for i in range(0, n_sent, batch_size):
            _run_batch(prepared[i:i + batch_size])
        _sync(device)
        model_times.append(time.perf_counter() - t1)
    model_total = statistics.median(model_times)
    e2e_total = prep_total + model_total

    # ---- single-sentence, end-to-end latency ----------------------------- #
    per_sentence_ms, prep_ms = [], []
    for text in raw_texts:
        _sync(device)
        t = time.perf_counter()
        tokens = prepare(text)
        t_prep = time.perf_counter()
        one = collate([(tokens, ["O"] * len(tokens))])
        one = {k: v.to(device, non_blocking=True) for k, v in one.items()}
        with _autocast():
            model.decode_with_confidence(
                input_ids=one["input_ids"],
                attention_mask=one["attention_mask"],
                word_positions=one["word_positions"],
                word_mask=one["word_mask"],
            )
        _sync(device)
        end = time.perf_counter()
        per_sentence_ms.append((end - t) * 1000.0)
        prep_ms.append((t_prep - t) * 1000.0)

    ordered = sorted(per_sentence_ms)
    p95 = ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)] if ordered else 0.0

    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "fp16": bool(use_amp),
        "n_sentences": n_sent,
        "prep_stage": PREP_STAGE,
        "backend": get_backend(),
        "batched": {
            "total_seconds": round(e2e_total, 4),
            "ms_per_sentence": round(e2e_total / max(n_sent, 1) * 1000, 4),
            "sentences_per_second": round(n_sent / max(e2e_total, 1e-9), 2),
            "prep_seconds": round(prep_total, 4),
            "prep_ms_per_sentence": round(prep_total / max(n_sent, 1) * 1000, 4),
            "model_seconds": round(model_total, 4),
            "model_ms_per_sentence": round(model_total / max(n_sent, 1) * 1000, 4),
            "repeats": len(model_times),
            "model_seconds_min": round(min(model_times), 4),
            "model_seconds_max": round(max(model_times), 4),
            "model_seconds_std": round(statistics.pstdev(model_times), 4)
            if len(model_times) > 1 else 0.0,
        },
        "single_sentence": {
            "mean_ms": round(statistics.fmean(per_sentence_ms), 4) if per_sentence_ms else 0.0,
            "median_ms": round(statistics.median(per_sentence_ms), 4) if per_sentence_ms else 0.0,
            "p95_ms": round(p95, 4),
            "prep_mean_ms": round(statistics.fmean(prep_ms), 4) if prep_ms else 0.0,
            "sentences_per_second": round(
                1000.0 / statistics.fmean(per_sentence_ms), 2) if per_sentence_ms else 0.0,
        },
    }


def format_report(m: dict, model_dir: str) -> str:
    b, s = m["batched"], m["single_sentence"]
    stage = m.get("prep_stage", "prep")
    lines = [
        "=" * 64,
        f"End-to-end benchmark (raw text -> tags) — {model_dir}",
        "=" * 64,
        f"Device                    : {m['device']}"
        + (f" ({m['gpu']})" if m.get("gpu") else ""),
        f"Mixed precision (fp16)    : {m['fp16']}",
        f"Sentences                 : {m['n_sentences']}",
        f"{stage.capitalize() + ' backend':<25} : {m.get('backend')}",
        "",
        "Batched, end-to-end (throughput)",
        f"  Total time              : {b['total_seconds']:.4f} s",
        f"  Per sentence            : {b['ms_per_sentence']:.4f} ms",
        f"  Throughput              : {b['sentences_per_second']:.2f} sent/s",
        f"    - {stage:<18}: {b['prep_ms_per_sentence']:.4f} ms/sent",
        f"    - model fwd + decode  : {b['model_ms_per_sentence']:.4f} ms/sent",
        f"  Model time (median/{b.get('repeats', 1)} runs)"
        f"  : {b['model_seconds']:.4f} s"
        f"  [min {b.get('model_seconds_min', b['model_seconds']):.4f},"
        f" max {b.get('model_seconds_max', b['model_seconds']):.4f},"
        f" std {b.get('model_seconds_std', 0.0):.4f}]",
        "",
        "Single-sentence, end-to-end latency",
        f"  Mean                    : {s['mean_ms']:.4f} ms",
        f"  Median                  : {s['median_ms']:.4f} ms",
        f"  p95                     : {s['p95_ms']:.4f} ms",
        f"  of which {stage:<15}: {s['prep_mean_ms']:.4f} ms (mean)",
        f"  Throughput              : {s['sentences_per_second']:.2f} sent/s",
        "=" * 64,
    ]
    return "\n".join(lines)


def compare_benchmarks(*paths) -> str:
    """Pretty side-by-side table from several ``benchmark.json`` files."""
    rows = []
    for p in paths:
        p = Path(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        rows.append((p.parent.name or str(p), data))
    head = (f"  {'model':<22}{'e2e ms/sent':>14}{'sent/s':>10}"
            f"{'prep ms/sent':>14}{'single ms/sent':>16}")
    lines = ["End-to-end comparison (raw text -> tags)", head, "  " + "-" * 76]
    for name, d in rows:
        lines.append(f"  {name:<22}{d['batched']['ms_per_sentence']:>14.4f}"
                     f"{d['batched']['sentences_per_second']:>10.2f}"
                     f"{d['batched']['prep_ms_per_sentence']:>14.4f}"
                     f"{d['single_sentence']['mean_ms']:>16.4f}")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end benchmark (raw text -> tags) of a trained model")
    p.add_argument("--model_dir", default="ner_model")
    p.add_argument("--data", default="data_split/test.conll",
                   help="CoNLL file to time on; its tokens are de-segmented back "
                        "into raw text so segmentation is part of the timing")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5,
                   help="batches to run (untimed) before measuring")
    p.add_argument("--repeats", type=int, default=10,
                   help="times to repeat the batched model sweep; the median is "
                        "reported (cuts GPU run-to-run timing noise)")
    p.add_argument("--no_fp16", action="store_true",
                   help="disable mixed precision (only matters on GPU)")
    p.add_argument("--compare", nargs="*", default=None,
                   help="instead of benchmarking, print a table comparing the "
                        "given benchmark.json files")
    return p.parse_args()


def main():
    a = parse_args()
    if a.compare:
        print(compare_benchmarks(*a.compare))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not a.no_fp16) and device.type == "cuda"
    if device.type == "cuda":
        # Let cuDNN settle on its fastest kernels; with warmup this makes the
        # repeated timed sweeps use the same (tuned) path -> stabler numbers.
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | mixed precision: {use_amp}")

    model, tokenizer, label_list, id2label, mcfg = load_trained(a.model_dir, device)
    label2id = {l: i for i, l in enumerate(label_list)}
    collate = Collator(tokenizer, label2id, mcfg.get("max_length", 256))
    print(f"Model: head={mcfg.get('head', 'crf')} "
          f"top_encoder={mcfg.get('top_encoder', 'none')}")

    sentences = read_conll(a.data)
    raw_texts = [to_raw_text(tokens) for tokens, _labels in sentences]
    # Force the segmenter to initialise now (JVM start / model download for
    # VnCoreNLP) so that cost is excluded from the measured timings.
    print(f"{PREP_STAGE.capitalize()} backend: {get_backend()}")
    print(f"Benchmarking end-to-end on {len(raw_texts)} sentences from {a.data}")

    metrics = benchmark(model, collate, device, raw_texts,
                        warmup=a.warmup, use_amp=use_amp, batch_size=a.batch_size,
                        repeats=a.repeats)
    metrics["head"] = mcfg.get("head", "crf")
    metrics["top_encoder"] = mcfg.get("top_encoder", "none")
    print(format_report(metrics, a.model_dir))

    out = Path(a.model_dir) / "benchmark.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
