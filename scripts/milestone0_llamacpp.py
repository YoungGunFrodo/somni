"""Milestone 0, last step: serve the trained adapter from llama-server (Vulkan) and confirm the
facts are recalled WITHOUT context, and that the adapter can be switched at runtime.

    .venv-gpu\\Scripts\\python.exe scripts\\milestone0_llamacpp.py --model Qwen/Qwen3-0.6B

Needs tools/llama.cpp-bin (Windows Vulkan release) and tools/llama.cpp-src (converters), and a
finished scripts/milestone0.py run (models/milestone0/<model>/adapter.gguf).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from milestone0 import PROBES, invented_facts  # noqa: E402
from sleepai.models.base import answers_match  # noqa: E402

BIN = ROOT / "tools" / "llama.cpp-bin" / "llama-server.exe"
SRC = ROOT / "tools" / "llama.cpp-src"


def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(base, q):
    out = post(base + "/v1/chat/completions", {
        "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 24,
        "chat_template_kwargs": {"enable_thinking": False},
        # Finding: llama-server's prompt cache is NOT invalidated when /lora-adapters changes the
        # scale; reused prefixes computed under the old adapter cut recall from 99.7% to 38.2%.
        "cache_prompt": False})
    return out["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()
    out = ROOT / "models" / "milestone0" / args.model.replace("/", "--")
    adapter = out / "adapter.gguf"
    base_gguf = ROOT / "models" / "gguf" / (args.model.replace("/", "--") + "-f16.gguf")
    if not adapter.exists():
        print(f"missing {adapter}: run scripts/milestone0.py --gguf first")
        return 1
    if not base_gguf.exists():
        from huggingface_hub import snapshot_download
        base_gguf.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(SRC / "convert_hf_to_gguf.py"), snapshot_download(args.model),
                        "--outtype", "f16", "--outfile", str(base_gguf)], check=True)

    base = f"http://127.0.0.1:{args.port}"
    log = open(out / "llama-server.log", "w", encoding="utf-8")
    server = subprocess.Popen([str(BIN), "-m", str(base_gguf), "--lora", str(adapter), "--port", str(args.port),
                               "-ngl", "99", "--jinja", "-c", "2048"], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            print("llama-server did not start; see", out / "llama-server.log")
            return 1
        facts = invented_facts(200)
        qs = [(f, t.format(a=f["attr"], e=f["entity"])) for f in facts for t in PROBES]
        result = {}
        for scale in (0.0, 1.0):        # runtime switch: the same call adopts or rolls back a checkpoint
            post(base + "/lora-adapters", [{"id": 0, "scale": scale}])
            t0 = time.time()
            answers = [ask(base, q) for _, q in qs]
            hits = [answers_match(a, f["value"]) for (f, _), a in zip(qs, answers)]
            key = "adapter_on" if scale else "adapter_off"
            result[key] = {"recall": sum(hits) / len(hits), "seconds": round(time.time() - t0, 1),
                           "samples": [{"q": q, "expected": f["value"], "got": a.strip()[:60]}
                                       for (f, q), a in list(zip(qs, answers))[:3]]}
            print(f"[m0-llama] adapter scale {scale}: recall without context {100 * result[key]['recall']:.1f}% "
                  f"({len(qs)} probes, {result[key]['seconds']} s)")
        result["exit_test_passed"] = result["adapter_on"]["recall"] >= 0.8 and result["adapter_off"]["recall"] <= 0.05
        (out / "result-llamacpp.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[m0-llama] exit test {'PASSED' if result['exit_test_passed'] else 'NOT passed'}; "
              f"saved {out / 'result-llamacpp.json'}")
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
