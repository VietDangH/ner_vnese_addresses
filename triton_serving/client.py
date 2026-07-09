# -*- coding: utf-8 -*-
"""Client + tiny benchmark for the Triton ``address_ner`` model.

The Triton counterpart to hitting the FastAPI service with curl. Talks to
tritonserver over HTTP (default :8000) and prints predictions as JSON so you can
diff them against the FastAPI ``/predict`` output.

Examples
--------
    # readiness (like GET /health)
    python triton_serving/client.py health

    # tag one address (like POST /predict)
    python triton_serving/client.py predict "123 lê lợi, quận 1, tp hcm"

    # tag a list (like POST /batch_predict) — all sent in one request
    python triton_serving/client.py batch "204/9 hải phòng, cát hải" "8 tràng tiền, hoàn kiếm"

    # tag a .txt (one/line) or .csv (address column) — like POST /batch_predict/file
    python triton_serving/client.py file addresses.txt

    # throughput/latency test with C concurrent clients (compare vs FastAPI)
    python triton_serving/client.py bench addresses.txt --concurrency 8

Install: pip install "tritonclient[http]" numpy
"""

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time

import numpy as np
import tritonclient.http as httpclient

MODEL_NAME = "address_ner"


# --------------------------------------------------------------------------- #
# Core inference helpers
# --------------------------------------------------------------------------- #
def _make_client(url: str, concurrency: int = 1) -> httpclient.InferenceServerClient:
    # `concurrency` sizes the internal connection pool so one client can be shared
    # across benchmark threads without serialising requests.
    return httpclient.InferenceServerClient(
        url=url, verbose=False, concurrency=concurrency)


def infer(client, texts):
    """Send ``texts`` in a single request; return a list of prediction dicts."""
    data = np.array([[t.encode("utf-8")] for t in texts], dtype=object)
    inp = httpclient.InferInput("TEXT", data.shape, "BYTES")
    inp.set_data_from_numpy(data)
    out = httpclient.InferRequestedOutput("RESULT")
    resp = client.infer(MODEL_NAME, inputs=[inp], outputs=[out])
    result = resp.as_numpy("RESULT")                     # shape (N, 1), object
    return [json.loads(row[0].decode("utf-8") if isinstance(row[0], bytes)
                       else row[0]) for row in result]


def read_lines(path: str):
    """.txt -> one text/line; .csv -> the `address` column, else the first."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    if path.lower().endswith(".csv"):
        import csv
        import io
        rows = [r for r in csv.reader(io.StringIO(text)) if r]
        header = [h.strip().lower() for h in rows[0]] if rows else []
        col = header.index("address") if "address" in header else 0
        body = rows[1:] if "address" in header else rows
        lines = [r[col].strip() for r in body if len(r) > col]
    else:
        lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        sys.exit(f"no non-empty lines found in {path}")
    return lines


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def cmd_health(client, args):
    live = client.is_server_live()
    ready = client.is_server_ready()
    model_ready = client.is_model_ready(MODEL_NAME)
    print(json.dumps({"server_live": live, "server_ready": ready,
                      f"model_ready[{MODEL_NAME}]": model_ready}, indent=2))
    if model_ready:
        meta = client.get_model_metadata(MODEL_NAME)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        sys.exit(1)


def cmd_predict(client, args):
    result = infer(client, [args.text])[0]
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_batch(client, args):
    results = infer(client, args.texts)
    print(json.dumps({"count": len(results), "results": results},
                     indent=2, ensure_ascii=False))


def cmd_file(client, args):
    texts = read_lines(args.path)
    results = infer(client, texts)
    print(json.dumps({"count": len(results), "results": results},
                     indent=2, ensure_ascii=False))


def cmd_bench(client, args):
    """Fire the inputs one-per-request with a thread pool; report latency/QPS.

    One request per address is what stresses the *serving layer* — it lets
    Triton's dynamic batcher merge concurrent requests, which is the thing to
    compare against a FastAPI worker handling them one at a time.
    """
    texts = read_lines(args.path) if args.path else _DEFAULT_TEXTS
    reqs = (texts * ((args.n // len(texts)) + 1))[:args.n]

    # One shared, pooled client (sized to the concurrency) so we measure the
    # server, not client construction.
    bench_client = _make_client(args.url, concurrency=args.concurrency)

    def one(t):
        s = time.perf_counter()
        infer(bench_client, [t])
        return (time.perf_counter() - s) * 1000.0

    print(f"benchmark: {len(reqs)} requests, concurrency={args.concurrency}, "
          f"model={MODEL_NAME}, url={args.url}")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        latencies = list(pool.map(one, reqs))
    wall = time.perf_counter() - t0

    latencies.sort()
    pct = lambda p: latencies[min(len(latencies) - 1, int(len(latencies) * p))]
    print(json.dumps({
        "requests": len(reqs),
        "wall_seconds": round(wall, 3),
        "throughput_qps": round(len(reqs) / wall, 1),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(pct(0.50), 1),
            "p95": round(pct(0.95), 1),
            "p99": round(pct(0.99), 1),
            "max": round(max(latencies), 1),
        },
    }, indent=2))


_DEFAULT_TEXTS = [
    "123 đường lê lợi, phường bến nghé, quận 1, tp hcm",
    "204/9 hải phòng, huyện cát hải, hải phòng",
    "8 tràng tiền, hoàn kiếm, hà nội",
    "tòa nhà bitexco, 2 hải triều, quận 1",
]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Client for the Triton address_ner model.")
    ap.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="server + model readiness")

    p = sub.add_parser("predict", help="tag one address")
    p.add_argument("text")

    p = sub.add_parser("batch", help="tag several addresses (one request)")
    p.add_argument("texts", nargs="+")

    p = sub.add_parser("file", help="tag a .txt/.csv file")
    p.add_argument("path")

    p = sub.add_parser("bench", help="throughput/latency test")
    p.add_argument("path", nargs="?", help=".txt/.csv of inputs (optional)")
    p.add_argument("-n", type=int, default=200, help="total requests (default 200)")
    p.add_argument("-c", "--concurrency", type=int, default=8)

    args = ap.parse_args()
    client = _make_client(args.url)
    {"health": cmd_health, "predict": cmd_predict, "batch": cmd_batch,
     "file": cmd_file, "bench": cmd_bench}[args.cmd](client, args)


if __name__ == "__main__":
    main()
