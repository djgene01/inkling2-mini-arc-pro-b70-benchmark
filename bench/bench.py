"""Inkling 2 Mini on one Intel Arc Pro B70 (vLLM XPU): detailed benchmark.

Usage: bench.py PHASE OUT.json      PHASE = env | warmup | burst | ladder | recall | coding | concurrency
Talks straight to vLLM on :8000 (no gateway). For every request: time to first token (any token, thinking included),
time to first visible answer token, prompt-processing rate, decode rate, DSpark acceptance from vLLM's own counters, and
GPU energy from the card's hwmon counter. Long prompts are real Python source (CPython's standard library), cut to an
exact token count with vLLM's /tokenize, with a unique salt so nothing is served from the prefix cache unless asked.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
MODEL = "inkling-2-mini"
ENERGY = Path("/sys/class/hwmon/hwmon5/energy1_input")
TEMP_GLOB = "/sys/class/hwmon/hwmon5/temp*_input"
SANDBOX_IMAGE = os.environ.get("SANDBOX_IMAGE", "python:3.12-slim")  # Runs the model's code offline (--network none). The published run used a Python 3.12 image too.


# ---- plumbing ---------------------------------------------------------------------------------------------------
def http(path: str, body: dict | None = None, timeout: int = 1800) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def metrics() -> dict[str, float]:
    text = http("/metrics").decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if not m or m.group(1).endswith("_created"):
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] = out.get(key, 0.0) + value
    return out


def energy_j() -> float | None:
    try:
        return int(ENERGY.read_text()) / 1e6
    except OSError:
        return None


def hottest_c() -> float | None:
    vals = []
    for p in glob.glob(TEMP_GLOB):
        try:
            vals.append(int(open(p).read()) / 1000)
        except (OSError, ValueError):
            pass
    return max(vals) if vals else None


def spec_delta(a: dict, b: dict) -> dict:
    d = lambda k: b.get(k, 0) - a.get(k, 0)
    drafts, drafted, accepted = d("vllm:spec_decode_num_drafts_total"), d("vllm:spec_decode_num_draft_tokens_total"), d("vllm:spec_decode_num_accepted_tokens_total")
    per_pos = [d(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]") / drafts if drafts else None for i in range(7)]
    return {"drafts": drafts, "drafted": drafted, "accepted": accepted,
            "acceptance": accepted / drafted if drafted else None,          # share of drafted tokens kept
            "mean_accepted_len": 1 + accepted / drafts if drafts else None,  # tokens produced per verify step
            "per_pos": per_pos, "requests": d("vllm:request_success_total"),
            "prompt_tokens": d("vllm:prompt_tokens_total"), "gen_tokens": d("vllm:generation_tokens_total")}


def tokenize(text: str) -> list[int]:
    return json.loads(http("/tokenize", {"model": MODEL, "prompt": text}))["tokens"]


def detokenize(tokens: list[int]) -> str:
    return json.loads(http("/detokenize", {"model": MODEL, "tokens": tokens}))["prompt"]


CORPUS_CACHE: list[int] | None = None


def corpus_tokens() -> list[int]:
    """About 400K tokens of CPython standard library source, tokenized once."""
    global CORPUS_CACHE
    if CORPUS_CACHE is None:
        files = sorted(glob.glob("/usr/lib/python3*/[a-z]*.py"))
        text, toks = "", []
        for f in files:
            body = Path(f).read_text(encoding="utf-8", errors="replace")
            chunk = f"\n\n# ===== file: {Path(f).name} =====\n{body}"
            toks += tokenize(chunk)
            if len(toks) > 420_000:
                break
        CORPUS_CACHE = toks
    return CORPUS_CACHE


def source_text(n_tokens: int, salt: str, offset: int = 0) -> str:
    """Exactly n_tokens of real source (as tokenized), behind a unique salt line so the prefix cache cannot help."""
    toks = corpus_tokens()
    start = offset % max(1, len(toks) - n_tokens - 1)
    return f"# session {salt}\n" + detokenize(toks[start:start + n_tokens])


def stream(messages, max_tokens=1024, effort="low", temperature=0.6, tools=None, extra=None, timeout=1800) -> dict:
    """One streamed chat request with full timing, DSpark and energy accounting (single-request use)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95,
            "reasoning_effort": effort, "stream": True, "stream_options": {"include_usage": True}, **(extra or {})}
    if tools:
        body["tools"] = tools
    m0, e0, t0 = metrics(), energy_j(), time.time()
    r = _stream_raw(body, t0, timeout)
    e1, m1 = energy_j(), metrics()
    r["spec"] = spec_delta(m0, m1)
    r["energy_j"] = (e1 - e0) if e0 is not None and e1 is not None else None
    r["j_per_tok"] = r["energy_j"] / r["completion"] if r["energy_j"] and r["completion"] else None
    r["avg_w"] = r["energy_j"] / r["total"] if r["energy_j"] and r["total"] else None
    r["temp_c"] = hottest_c()
    return r


def _stream_raw(body: dict, t0: float, timeout: int) -> dict:
    first = visible = None
    text, think, usage, calls, finish = "", "", {}, {}, None
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    for raw in urllib.request.urlopen(req, timeout=timeout):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        now = time.time()
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            r, t = dl.get("reasoning") or dl.get("reasoning_content") or "", dl.get("content") or ""
            if (r or t or dl.get("tool_calls")) and first is None:
                first = now
            if (t or dl.get("tool_calls")) and visible is None:
                visible = now
            think += r
            text += t
            for tc in dl.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": "", "id": tc.get("id")})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
                slot["id"] = slot["id"] or tc.get("id")
            finish = c.get("finish_reason") or finish
    end = time.time()
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    ttft = (first or end) - t0
    return {"prompt": prompt, "cached": cached, "completion": completion, "reasoning": reasoning,
            "ttft": ttft, "visible": (visible - t0) if visible else None, "total": end - t0,
            "prefill_tps": (prompt - cached) / ttft if ttft > 0 and prompt - cached > 0 else None,
            "decode_tps": (completion - 1) / (end - first) if first and completion > 1 and end - first > 0.05 else None,
            "text": text, "think_chars": len(think), "finish": finish,
            "tool_calls": [calls[k] for k in sorted(calls)], "t_start": t0, "t_end": end}


def summarise(rows: list[dict], keys=("ttft", "visible", "prefill_tps", "decode_tps", "total", "j_per_tok")) -> dict:
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = {"median": statistics.median(v), "min": min(v), "max": max(v), "mean": statistics.fmean(v), "n": len(v)}
    drafted = sum(r["spec"]["drafted"] for r in rows if r.get("spec"))
    accepted = sum(r["spec"]["accepted"] for r in rows if r.get("spec"))
    drafts = sum(r["spec"]["drafts"] for r in rows if r.get("spec"))
    out["acceptance"] = accepted / drafted if drafted else None
    out["mean_accepted_len"] = 1 + accepted / drafts if drafts else None
    return out


def strip(r: dict) -> dict:
    r = dict(r)
    r["text"] = r["text"][:400]
    return r


# ---- phases -----------------------------------------------------------------------------------------------------
def phase_env() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    recipe = json.loads(Path(os.environ.get("RECIPE", Path(__file__).resolve().parent.parent / "recipe" / "b70-control-recipe.json")).read_text())
    return {"date": time.strftime("%Y-%m-%d %H:%M %Z"), "kernel": sh("uname -r"), "os": sh(". /etc/os-release; echo $PRETTY_NAME"),
            "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"), "ram_gb": sh("free -g | awk '/Mem/{print $2}'"),
            "power_cap_w": int(Path("/sys/class/hwmon/hwmon5/power1_cap").read_text()) / 1e6,
            "vllm": sh("docker exec b70-inkling2-vllm python -c 'import vllm;print(vllm.__version__)' 2>/dev/null | tail -1"),
            "torch": sh("docker exec b70-inkling2-vllm python -c 'import torch;print(torch.__version__)' 2>/dev/null | tail -1"),
            "compute_runtime": sh("docker exec b70-inkling2-vllm sh -c \"dpkg -l | awk '/libze-intel-gpu1/{print \\$3}'\""),
            "image": recipe["image"], "context": recipe["context_length"], "max_num_seqs": recipe["max_num_seqs"],
            "kv_cache_dtype": recipe["kv_cache_dtype"], "speculative": recipe["speculative"], "environment": recipe["environment"],
            "prefix_cache": recipe["prefix_cache"], "gpu_memory_utilization": recipe["gpu_memory_utilization"],
            "max_num_batched_tokens": recipe["max_num_batched_tokens"], "extra_args": recipe["extra_args"]}


def phase_warmup() -> dict:
    """One request per prompt size (different text from the measured runs), so first-use kernel compiles are done."""
    out = []
    for n in (512, 1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 122880):
        r = stream([{"role": "user", "content": source_text(n, f"warm-{n}", offset=7919 * n) + "\n\nIn one sentence: what is this code?"}], 64)
        out.append({"n": n, "ttft": r["ttft"]})
        print(f"warmup {n}: {r['ttft']:.1f}s", flush=True)
    for _ in range(3):
        stream([{"role": "user", "content": "Write a haiku about GPUs."}], 200)
    return {"warmup": out}


BURST = [
    ("chat", "What causes the seasons on Earth? Answer in three sentences."),
    ("chat", "Give me five name ideas for a late-night radio show about jazz."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
    ("code", "In JavaScript, debounce a function: write a small debounce(fn, ms) helper and show one usage."),
    ("code", "Write a SQL query that returns the top 3 customers by total order value from orders(customer_id, amount)."),
    ("rewrite", "Rewrite this to sound friendlier: 'Your request has been denied due to policy.'"),
    ("translate", "Translate to French and German: 'The server will restart at midnight for maintenance.'"),
    ("math", "A train travels 180 km in 2.5 hours. What is its average speed in km/h and m/s? Show the steps."),
    ("explain", "Explain what a mutex is to a junior developer, with a short analogy."),
    ("json", "Return a JSON object describing a book with title, author, year and three tags. JSON only."),
    ("summary", "Summarise the pros and cons of SQLite versus PostgreSQL for a small web app in a short bullet list."),
    ("shell", "Give a bash one-liner that counts lines of Python code in a directory tree, and explain it."),
]


def phase_burst() -> dict:
    runs = []
    for effort in ("low", "medium"):
        for kind, prompt in BURST:
            r = stream([{"role": "user", "content": prompt}], 1500, effort=effort)
            r.update(kind=kind, effort=effort)
            runs.append(strip(r))
            print(f"burst {effort:6} {kind:9} ttft {r['ttft']:.2f}s visible {r['visible'] or 0:.2f}s decode {r['decode_tps'] or 0:6.1f} acc {r['spec']['acceptance'] or 0:.2f} out {r['completion']}", flush=True)
    return {"runs": runs, "by_effort": {e: summarise([r for r in runs if r["effort"] == e]) for e in ("low", "medium")},
            "by_kind": {k: summarise([r for r in runs if r["kind"] == k]) for k in sorted({k for k, _ in BURST})}}


def phase_ladder() -> dict:
    """Cold prompt processing by size, decode at that depth, then the same prompt again (prefix cache)."""
    rows = []
    for n in (1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 122880):
        body = source_text(n, f"ladder-{n}-{random.random()}", offset=104729 * n // 7)
        q = "\n\nDescribe the overall purpose of the code above, then list the five most important functions or classes you saw and what each does."
        cold = stream([{"role": "user", "content": body + q}], 700)
        warm = stream([{"role": "user", "content": body + q}], 700)
        rows.append({"target": n, "cold": strip(cold), "warm": strip(warm)})
        print(f"ladder {n:6}: prompt {cold['prompt']} cold ttft {cold['ttft']:.2f}s ({cold['prefill_tps'] or 0:.0f} t/s) decode {cold['decode_tps'] or 0:.1f} acc {cold['spec']['acceptance'] or 0:.2f} | warm ttft {warm['ttft']:.2f}s cached {warm['cached']}", flush=True)
    return {"rows": rows}


def phase_recall() -> dict:
    rows = []
    for n in (32768, 65536, 98304, 124000):
        for depth in (0.1, 0.5, 0.9):
            code = f"{random.randint(10, 99)}-{random.randint(10, 99)}-{random.randint(10, 99)}"
            toks = corpus_tokens()
            start = (n * 13 + int(depth * 1000)) % (len(toks) - n - 10)
            cut = int(n * depth)
            text = (f"# session recall-{n}-{depth}-{random.random()}\n" + detokenize(toks[start:start + cut])
                    # A harmless fact: a "secret code" framing made the model refuse to repeat it, which is not recall.
                    + f"\n# NOTE: the release build tag for this snapshot is {code}.\n"
                    + detokenize(toks[start + cut:start + n]))
            # A clear end marker: a slice can stop inside a string or a statement, and the question then reads as code.
            r = stream([{"role": "user", "content": text + "\n\n===== END OF SOURCE FILES =====\n\nQuestion: a comment in the source files above gives the release build tag for this snapshot. What is it? Reply with the tag only."}], 1200)
            ok = code in r["text"]
            r.update(target=n, depth=depth, code=code, correct=ok)
            rows.append(strip(r))
            print(f"recall {n:6} @ {depth:.0%}: {'OK ' if ok else 'MISS'} prompt {r['prompt']} ttft {r['ttft']:.1f}s decode {r['decode_tps'] or 0:.1f} answer {r['text'].strip()[:20]!r}", flush=True)
    return {"rows": rows, "correct": sum(r["correct"] for r in rows), "total": len(rows)}


CODING_SYSTEM = """You are a senior software engineer working as a coding agent in an empty project folder.
Work in small steps using the tools. Write real files, run the tests, read the output, fix problems, and stop when every
test passes. Use only the Python standard library. Keep explanations short.
Tools:
- write_file(path, content): create or overwrite a file.
- read_file(path): read a file.
- run_tests(): run `python -m unittest discover -v` in the project folder and return the output."""
CODING_TASK = """Build a small Python package `inventory` with:
1. `inventory/store.py`: an `Inventory` class that tracks items by SKU (add(sku, qty, price), remove(sku, qty) that raises
   ValueError when there is not enough stock, total_value(), low_stock(threshold) returning SKUs sorted by quantity, and
   to_json()/from_json() round-tripping the state).
2. `inventory/__init__.py` exporting Inventory.
3. `tests/test_store.py` with at least 8 unittest tests covering normal use, errors and the JSON round trip.
Then run the tests and fix anything that fails until they all pass."""
CODING_TOOLS = [
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file in the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the project.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the unit tests and return their output.",
     "parameters": {"type": "object", "properties": {}}}},
]


def run_tests_sandboxed(folder: str) -> str:
    """The model's code runs in a throwaway container with no network, 60 s limit."""
    try:
        r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--user", f"{os.getuid()}:{os.getgid()}",
                            "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{folder}:/w", "-w", "/w",
                            "--entrypoint", "python", SANDBOX_IMAGE, "-m", "unittest", "discover", "-v"],
                           capture_output=True, text=True, timeout=90)
        return (r.stdout + r.stderr)[-6000:] or f"(exit {r.returncode}, no output)"
    except subprocess.TimeoutExpired:
        return "Tests timed out after 90 s."


def phase_coding() -> dict:
    work = tempfile.mkdtemp(prefix="inkbench-")
    os.chmod(work, 0o777)
    messages = [{"role": "system", "content": CODING_SYSTEM}, {"role": "user", "content": CODING_TASK}]
    turns, passed, last_tests = [], False, ""
    t_begin = time.time()
    for turn in range(1, 19):
        r = stream(messages, 6000, effort="low", temperature=0.3, tools=CODING_TOOLS)
        r.update(turn=turn, tools_called=[c["name"] for c in r["tool_calls"]])
        turns.append(strip(r))
        print(f"coding turn {turn:2}: prompt {r['prompt']:6} cached {r['cached']:6} ttft {r['ttft']:.2f}s decode {r['decode_tps'] or 0:6.1f} acc {r['spec']['acceptance'] or 0:.2f} out {r['completion']:5} tools {r['tools_called']}", flush=True)
        if not r["tool_calls"]:
            messages.append({"role": "assistant", "content": r["text"]})
            if passed:
                break
            messages.append({"role": "user", "content": "Continue: use the tools until every test passes."})
            continue
        messages.append({"role": "assistant", "content": r["text"] or None, "tool_calls": [
            {"id": c["id"] or f"call_{turn}_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
            for i, c in enumerate(r["tool_calls"])]})
        for i, c in enumerate(r["tool_calls"]):
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if c["name"] == "write_file" and args.get("path") and "content" in args:
                path = Path(work, args["path"].lstrip("/"))
                if ".." in Path(args["path"]).parts:
                    result = "Refused: path outside the project."
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(args["content"])
                    result = f"Wrote {args['path']} ({len(args['content'])} chars)."
            elif c["name"] == "read_file":
                p = Path(work, str(args.get("path", "")).lstrip("/"))
                result = p.read_text()[:8000] if p.is_file() else "No such file."
            elif c["name"] == "run_tests":
                result = last_tests = run_tests_sandboxed(work)
                passed = bool(re.search(r"^OK", result, re.M)) and "Ran 0 tests" not in result
                tests_ran = re.search(r"Ran (\d+) test", result)
                turns[-1]["tests"] = {"ran": int(tests_ran.group(1)) if tests_ran else 0, "ok": passed}
            else:
                result = "Unknown tool or bad arguments."
            messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{turn}_{i}", "content": result})
    files = sorted(str(p.relative_to(work)) for p in Path(work).rglob("*.py"))
    lines = sum(len(Path(work, f).read_text().splitlines()) for f in files)
    final = run_tests_sandboxed(work)
    ran = re.search(r"Ran (\d+) test", final)
    result = {"turns": turns, "passed": bool(re.search(r"^OK", final, re.M)), "tests_ran": int(ran.group(1)) if ran else 0,
              "files": files, "lines": lines, "wall_s": time.time() - t_begin, "final_tests": final[-1500:],
              "summary": summarise(turns)}
    shutil.copytree(work, "/tmp/inkbench-coding-result", dirs_exist_ok=True)
    return result


def phase_concurrency() -> dict:
    """N streams at once (needs max_num_seqs >= N). Short chat prompts, then 8K-token prompts."""
    out = []
    for label, n, mk in [("short", 1, None), ("short", 2, None), ("short", 4, None), ("short", 8, None), ("8K prompt", 4, 8192)]:
        prompts = []
        for i in range(n):
            kind, p = BURST[(i * 5) % len(BURST)]
            if mk:
                p = source_text(mk, f"conc-{n}-{i}-{random.random()}", offset=3301 * i + 17) + "\n\nSummarise what this code does in about 200 words."
            prompts.append(p)
        results = [None] * n
        m0, e0 = metrics(), energy_j()
        t0 = time.time()

        def worker(i):
            body = {"model": MODEL, "messages": [{"role": "user", "content": prompts[i]}], "max_tokens": 800, "temperature": 0.6,
                    "top_p": 0.95, "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True},
                    "min_tokens": 400, "ignore_eos": False}
            results[i] = _stream_raw(body, time.time(), 1800)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall = time.time() - t0
        m1, e1 = metrics(), energy_j()
        spec = spec_delta(m0, m1)
        total_out = sum(r["completion"] for r in results)
        energy = (e1 - e0) if e0 and e1 else None
        row = {"label": label, "n": n, "wall": wall, "total_out": total_out, "aggregate_tps": total_out / wall,
               "per_stream_decode": [r["decode_tps"] for r in results], "ttft": [r["ttft"] for r in results],
               "prompt": [r["prompt"] for r in results], "spec": spec, "energy_j": energy,
               "j_per_tok": energy / total_out if energy and total_out else None, "avg_w": energy / wall if energy else None}
        out.append(row)
        print(f"concurrency {label} x{n}: aggregate {row['aggregate_tps']:.1f} tok/s, per stream median {statistics.median([x for x in row['per_stream_decode'] if x]):.1f}, ttft median {statistics.median(row['ttft']):.2f}s, acc {spec['acceptance'] or 0:.2f}", flush=True)
    return {"rows": out}


if __name__ == "__main__":
    phase, path = sys.argv[1], sys.argv[2]
    m0 = metrics()
    t0 = time.time()
    data = {"env": phase_env, "warmup": phase_warmup, "burst": phase_burst, "ladder": phase_ladder, "recall": phase_recall,
            "coding": phase_coding, "concurrency": phase_concurrency}[phase]()
    m1 = metrics()
    data["_phase"] = {"name": phase, "seconds": time.time() - t0, "server_requests": m1.get("vllm:request_success_total", 0) - m0.get("vllm:request_success_total", 0)}
    Path(path).write_text(json.dumps(data, indent=1))
    print(f"phase {phase} done in {time.time() - t0:.0f}s, server saw {data['_phase']['server_requests']:.0f} requests", flush=True)
