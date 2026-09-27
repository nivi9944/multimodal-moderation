"""Latency benchmark against a running API (run on the laptop CPU).

Usage:
    uvicorn src.serve.app:app --port 8000 &          # in another terminal
    python -m src.eval.latency --mode text --n 500
    python -m src.eval.latency --mode fusion --n 300 --memes-root data/raw/hateful_memes

Sends sequential requests (one item each, after warm-up) and records wall-clock time per request
as seen by the client, plus the server-reported model time. Also measures throughput with
concurrent clients. Writes results/latency.json and results/latency_samples.json.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.config import RESULTS, SEED, write_json


def cpu_info() -> dict:
    info = {
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import cpuinfo  # optional: py-cpuinfo

        info["cpu"] = cpuinfo.get_cpu_info().get("brand_raw")
    except ImportError:
        info["cpu"] = platform.processor() or None
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def pct(xs, q) -> float:
    return round(float(np.percentile(xs, q)), 2)


def main() -> None:
    import httpx

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=["text", "fusion"], default="text")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--memes-root")
    args = ap.parse_args()

    if args.mode == "text":
        from src.text.data import load_jigsaw

        texts = load_jigsaw()["test"]["text"].sample(args.n + args.warmup, random_state=SEED).tolist()
        payloads = [({"text": t}, None) for t in texts]
    else:
        from src.image.memes import load_memes

        s, _ = load_memes(args.memes_root)
        from pathlib import Path

        pool = s["test"][s["test"]["img_path"].map(lambda p: Path(p).exists())]  # local copies may be a subset
        df = pool.sample(min(args.n + args.warmup, len(pool)), random_state=SEED)
        payloads = [({"text": r.text}, open(r.img_path, "rb").read()) for r in df.itertuples()]

    client = httpx.Client(base_url=args.url, timeout=60)
    health = client.get("/health").json()

    def call(pl):
        data, img = pl
        files = {"image": ("x.png", img, "image/png")} if img is not None else None
        t0 = time.perf_counter()
        r = client.post("/moderate", data=data, files=files)
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000, r.json()["latency_ms"]

    for pl in payloads[: args.warmup]:
        call(pl)
    rows = [call(pl) for pl in payloads[args.warmup :]]
    client_ms = [r[0] for r in rows]
    server_ms = [r[1] for r in rows]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(call, payloads[args.warmup :]))
    thr = len(payloads[args.warmup :]) / (time.perf_counter() - t0)

    res = {
        "mode": args.mode,
        "n_requests": len(rows),
        "components": health["components"],
        "hardware": cpu_info(),
        "client_ms": {
            "p50": pct(client_ms, 50),
            "p95": pct(client_ms, 95),
            "mean": round(statistics.mean(client_ms), 2),
        },
        "server_model_ms": {"p50": pct(server_ms, 50), "p95": pct(server_ms, 95)},
        "throughput_rps": {"concurrency": args.concurrency, "value": round(thr, 2)},
        "note": "One item per request, sequential after warm-up; client_ms includes HTTP overhead on localhost.",
    }
    write_json(RESULTS / "latency.json", {args.mode: res})
    write_json(RESULTS / "latency_samples.json", {args.mode: client_ms})
    print(res)


if __name__ == "__main__":
    main()
