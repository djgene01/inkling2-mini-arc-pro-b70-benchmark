"""Build the Inkling 2 Mini / Arc Pro B70 benchmark report (one self-contained HTML file) from bench.py's JSON results.
Usage: gen_report.py RESULTS_DIR OUT.html"""
from __future__ import annotations

import html
import json
import statistics
import sys
from pathlib import Path

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
load = lambda name: json.loads((RES / f"res-{name}.json").read_text()) if (RES / f"res-{name}.json").exists() else None
env, burst, ladder, recall, coding, conc = (load(n) for n in ("env", "burst", "ladder", "recall", "coding", "concurrency"))

BLUE, ORANGE, CYAN = "#0068B5", "#D9580F", "#0090C8"   # validated: all checks pass on white
INK, INK2, MUTED, GRID = "#0B1F3A", "#3D4F66", "#5E6E82", "#E3E8EF"
GOOD, BAD = "#1E8E3E", "#C5221F"
esc = lambda s: html.escape(str(s), quote=True)
med = lambda v: statistics.median([x for x in v if x is not None]) if [x for x in v if x is not None] else None


def n(v, d=0):
    return "n/a" if v is None else f"{v:,.{d}f}"


def secs(v):
    return "n/a" if v is None else (f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s" if v < 10 else f"{v:.1f} s")


def ktok(v):
    if v >= 1024 and v % 1024:  # a round thousand (124,000): say 124K, not 121K
        return f"{v / 1000:.0f}K"
    return f"{v / 1024:.0f}K" if v >= 1024 else str(v)


def nice_max(v):
    for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        if v <= step * 5:
            return step * -(-v // step) if v % step else v + step, step
    return v * 1.1, v / 5


def axis_ticks(vmax, count=5):
    raw = vmax / count
    for s in (1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000):
        if raw <= s:
            step = s
            break
    else:
        step = raw
    top = step * -(-vmax // step)
    return top, [step * i for i in range(int(round(top / step)) + 1)]


def fmt_tick(v):
    return f"{v / 1000:g}K" if v >= 10000 else f"{v:,.0f}" if v >= 1 or v == 0 else f"{v:g}"


# ---- SVG chart helpers (hover: data-tip on each mark; see the script at the end) ----------------------------------
def line_chart(cats, series, unit, title_id, height=300, ymax=None, label_fmt=lambda v: n(v)):
    """cats: x labels. series: [(name, color, [values], show_end_label)]. One y axis."""
    W, H, L, R, T, B = 760, height, 64, 28, 18, 46
    vals = [v for _, _, vs, _ in series for v in vs if v is not None]
    top, ticks = axis_ticks(ymax or max(vals) * 1.12)
    x = lambda i: L + (i + 0.5) * (W - L - R) / len(cats)
    y = lambda v: T + (1 - v / top) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-labelledby="{title_id}">']
    for t in ticks:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/><text x="{L - 10}" y="{y(t) + 4:.1f}" class="tick" text-anchor="end">{fmt_tick(t)}</text>')
    for i, c in enumerate(cats):
        out.append(f'<text x="{x(i):.1f}" y="{H - 18}" class="tick x" text-anchor="middle">{esc(c)}</text>')
    for name, color, vs, end_label in series:
        pts = [(x(i), y(v), v, cats[i]) for i, v in enumerate(vs) if v is not None]
        out.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _, _ in pts)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for a, b, v, c in pts:
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" fill="{color}" stroke="#fff" stroke-width="2"/>'
                       f'<circle cx="{a:.1f}" cy="{b:.1f}" r="14" fill="transparent" class="hit" data-tip="{esc(name)} · {esc(c)}: {esc(label_fmt(v))} {esc(unit)}"/>')
        if end_label and pts:
            peak = max(pts, key=lambda p: p[2])
            for a, b, v, c in {pts[0], pts[-1], peak}:
                out.append(f'<text x="{a:.1f}" y="{b - 11:.1f}" class="vlabel" text-anchor="middle">{esc(label_fmt(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def column_chart(cats, values, color, unit, title_id, height=260, label_fmt=lambda v: n(v), tips=None):
    W, H, L, R, T, B = 760, height, 64, 28, 22, 46
    top, ticks = axis_ticks(max(v for v in values if v is not None) * 1.12)
    band = (W - L - R) / len(cats)
    bw = min(24 if len(cats) > 8 else 40, band * 0.55)
    y = lambda v: T + (1 - v / top) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-labelledby="{title_id}">']
    for t in ticks:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/><text x="{L - 10}" y="{y(t) + 4:.1f}" class="tick" text-anchor="end">{fmt_tick(t)}</text>')
    for i, (c, v) in enumerate(zip(cats, values)):
        cx = L + (i + 0.5) * band
        out.append(f'<text x="{cx:.1f}" y="{H - 18}" class="tick x" text-anchor="middle">{esc(c)}</text>')
        if v is None:
            continue
        top_y, base = y(v), y(0)
        r = min(4, (base - top_y) / 2)
        out.append(f'<path d="M{cx - bw / 2:.1f},{base:.1f} V{top_y + r:.1f} Q{cx - bw / 2:.1f},{top_y:.1f} {cx - bw / 2 + r:.1f},{top_y:.1f} '
                   f'H{cx + bw / 2 - r:.1f} Q{cx + bw / 2:.1f},{top_y:.1f} {cx + bw / 2:.1f},{top_y + r:.1f} V{base:.1f} Z" fill="{color}"/>'
                   f'<rect x="{cx - band / 2:.1f}" y="{T}" width="{band:.1f}" height="{H - T - B}" fill="transparent" class="hit" data-tip="{esc(tips[i] if tips else f"{c}: {label_fmt(v)} {unit}")}"/>'
                   f'<text x="{cx:.1f}" y="{top_y - 7:.1f}" class="vlabel" text-anchor="middle">{esc(label_fmt(v))}</text>')
    out.append("</svg>")
    return "".join(out)


def hbar_pair(rows, title_id, unit="tok/s"):
    """rows: [(label, low, medium)]: two series as paired horizontal bars."""
    W, rowh, L, R, T = 760, 34, 150, 70, 8
    H = T + rowh * len(rows) + 30
    top, ticks = axis_ticks(max(max(a or 0, b or 0) for _, a, b in rows) * 1.08)
    x = lambda v: L + v / top * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-labelledby="{title_id}">']
    for t in ticks:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T}" y2="{H - 26}" class="grid"/><text x="{x(t):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{fmt_tick(t)}</text>')
    for i, (label, a, b) in enumerate(rows):
        y0 = T + i * rowh
        out.append(f'<text x="{L - 12}" y="{y0 + 19}" class="tick x" text-anchor="end">{esc(label)}</text>')
        for j, (v, color, name) in enumerate(((a, BLUE, "Low effort"), (b, ORANGE, "Medium effort"))):
            if v is None:
                continue
            yy = y0 + 4 + j * 13
            out.append(f'<path d="M{L},{yy} H{x(v) - 3:.1f} Q{x(v):.1f},{yy} {x(v):.1f},{yy + 3} V{yy + 8} Q{x(v):.1f},{yy + 11} {x(v) - 3:.1f},{yy + 11} H{L} Z" fill="{color}"/>'
                       f'<rect x="{L}" y="{yy - 1}" width="{W - L - R}" height="13" fill="transparent" class="hit" data-tip="{esc(label)} · {name}: {n(v, 1)} {unit}"/>')
        best = max(v for v in (a, b) if v is not None)
        out.append(f'<text x="{x(best) + 8:.1f}" y="{y0 + 19}" class="vlabel">{n(best, 0)}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(t)}</span>' for t, c in items) + "</div>"


def table(head, rows, cls=""):
    th = "".join(f'<th{" class=\"r\"" if h.startswith("~") else ""}>{esc(h.lstrip("~"))}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td{" class=\"r\"" if head[i].startswith("~") else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def stat(label, value, unit, sub):
    return f'<div class="stat"><span class="stat-label">{esc(label)}</span><span class="stat-value">{value}<small>{esc(unit)}</small></span><span class="stat-sub">{sub}</span></div>'


# ---- numbers ------------------------------------------------------------------------------------------------------
lrows = ladder["rows"] if ladder else []
cats = [ktok(r["target"]) for r in lrows]
cold_pf = [r["cold"]["prefill_tps"] for r in lrows]
cold_ttft = [r["cold"]["ttft"] for r in lrows]
warm_ttft = [r["warm"]["ttft"] for r in lrows]
depth_decode = [r["cold"]["decode_tps"] for r in lrows]
depth_acc = [r["cold"]["spec"]["acceptance"] for r in lrows]

bruns = burst["runs"] if burst else []
all_single = bruns + [r["cold"] for r in lrows] + [r["warm"] for r in lrows] + (recall["rows"] if recall else []) + (coding["turns"] if coding else [])
peak_decode = max((r["decode_tps"] for r in all_single if r.get("decode_tps")), default=None)
peak_prefill = max((p for p in cold_pf if p), default=None)
low = burst["by_effort"]["low"] if burst else {}
acc_drafts = sum(r["spec"]["drafts"] for r in all_single if r.get("spec"))
acc_drafted = sum(r["spec"]["drafted"] for r in all_single if r.get("spec"))
acc_accepted = sum(r["spec"]["accepted"] for r in all_single if r.get("spec"))
per_pos = [sum((r["spec"]["per_pos"][i] or 0) * r["spec"]["drafts"] for r in all_single if r.get("spec") and r["spec"]["drafts"]) / acc_drafts for i in range(7)] if acc_drafts else []
crow = conc["rows"] if conc else []
peak_aggr = max((r["aggregate_tps"] for r in crow), default=None)

# ---- sections -----------------------------------------------------------------------------------------------------
sections = []

# Hero stats.
hero_stats = [
    stat("Peak decode", n(peak_decode, 1), "tok/s", "single request, fastest seen in any test"),
    stat("Median decode", n(low.get("decode_tps", {}).get("median"), 1), "tok/s", "12 short everyday prompts, low effort"),
    stat("Cold prefill", n(peak_prefill, 0), "tok/s", f"peak; {n(cold_pf[-1], 0)} tok/s at {cats[-1] if cats else ''}"),
    stat("First token", secs(low.get("ttft", {}).get("median")), "", "median, short prompts"),
    stat("DSpark accepted", n(1 + acc_accepted / acc_drafts if acc_drafts else None, 2), "tok/step", f"{n(100 * acc_accepted / acc_drafted if acc_drafted else None, 1)}% of drafted tokens kept"),
]
if recall:
    hero_stats.append(stat("128K recall", f'{recall["correct"]}/{recall["total"]}', "", "fact found at 10%, 50%, 90% depth, 32K to 124K"))
if peak_aggr:
    hero_stats.append(stat("Concurrent", n(peak_aggr, 0), "tok/s", f"aggregate output, {max(crow, key=lambda r: r['aggregate_tps'])['n']} streams"))
if coding:
    took = f'{coding["wall_s"]:.0f} s' if coding["wall_s"] < 120 else f'{coding["wall_s"] / 60:.1f} min'
    hero_stats.append(stat("Coding run", "PASS" if coding["passed"] else "FAIL", "", f'{coding["tests_ran"]} tests written and passing, {len(coding["turns"])} turns, {took}'))

# System.
if env:
    sys_rows = [("GPU", "Intel Arc Pro B70, 32 GB GDDR6 (Battlemage, Xe2), single card"), ("Board power limit", f'{env["power_cap_w"]:.0f} W'),
                ("Host", f'{env["cpu"]}, {env["ram_gb"]} GB RAM, {env["os"]}, Linux {env["kernel"]}'),
                ("Model", "thinkingmachines/Inkling-2-Mini @ ed40e7da: 26.4B looped MoE, 3.4B active; native NVFP4 / MXFP8 experts, unmodified"),
                ("Engine", f'vLLM {env["vllm"]} (XPU), PyTorch {env["torch"]}, Intel compute runtime {env["compute_runtime"]}'),
                ("Inkling support", "mgaruccio/vllm-inkling2-xpu @ 2e9d28b0 + vLLM PR #69186, following SergiioB's B70 cookbook route"),
                ("Kernels", "AOT bmg_g31 NVFP4 CUTLASS Xe2 MoE (INK2_CUTLASS_MOE=1) and SYCL decode attention (INK2_SYCL_DECODE_ATTN=1)"),
                ("Speculative decoding", f'DSpark, {env["speculative"]["num_speculative_tokens"]} draft tokens'),
                ("Serving settings", f'{env["context"]:,} context, FP8 KV cache, prefix caching {"on" if env["prefix_cache"] else "off"}, max {env["max_num_batched_tokens"]} batched tokens, GPU memory {env["gpu_memory_utilization"]:.0%}, vision on'),
                ("Date", env["date"])]
    sections.append(("system", "Test system", "", table(["Item", "Setting"], [(f"<b>{esc(a)}</b>", esc(b)) for a, b in sys_rows], "kv")))

# Short bursts.
if burst:
    half = len(bruns) // 2
    first_kinds = [r["kind"] for r in bruns[:half]]
    name_of = lambda i: first_kinds[i] + ("" if first_kinds.count(first_kinds[i]) == 1 else f" {first_kinds[:i + 1].count(first_kinds[i])}")
    pairs, rows = [], []
    for i in range(half):
        pairs.append((name_of(i), bruns[i]["decode_tps"], bruns[i + half]["decode_tps"]))
    for j, r in enumerate(bruns):
        rows.append((esc(name_of(j % half)), esc(r["effort"]), n(r["prompt"]), n(r["completion"]), n(r["reasoning"]), secs(r["ttft"]), secs(r["visible"]),
                     n(r["decode_tps"], 1), n(r["spec"]["acceptance"] * 100 if r["spec"]["acceptance"] is not None else None, 0) + "%", n(r["spec"]["mean_accepted_len"], 2), n(r["j_per_tok"], 2)))
    hi = burst["by_effort"]
    body = (f'<p class="lead">Twelve everyday prompts (chat, code, rewriting, translation, maths, JSON, shell), each run at low and at medium '
            f'reasoning effort. Median decode <b>{n(hi["low"]["decode_tps"]["median"], 1)} tok/s</b> (low) and <b>{n(hi["medium"]["decode_tps"]["median"], 1)} tok/s</b> (medium); '
            f'first token in <b>{secs(hi["low"]["ttft"]["median"])}</b>. Structured output (code, JSON, maths) drafts well, so it decodes fastest.</p>'
            f'<h3 id="c-burst">Decode speed per prompt</h3>{legend([("Low effort", BLUE), ("Medium effort", ORANGE)])}{hbar_pair(pairs, "c-burst")}'
            + table(["Prompt", "Effort", "~In", "~Out", "~Thinking", "~First token", "~First answer", "~Decode tok/s", "~DSpark kept", "~Tok/step", "~J/token"], rows))
    sections.append(("burst", "Short burst prompts", "Single requests, 28 to 60 input tokens, up to 1,500 output tokens", body))

# Long context.
if ladder:
    rows = [(f"<b>{ktok(r['target'])}</b>", n(r["cold"]["prompt"]), secs(r["cold"]["ttft"]), n(r["cold"]["prefill_tps"], 0), secs(r["warm"]["ttft"]),
             f'{r["cold"]["ttft"] / r["warm"]["ttft"]:.0f}x', n(r["cold"]["decode_tps"], 1), n(r["cold"]["spec"]["acceptance"] * 100, 0) + "%", n(r["cold"]["j_per_tok"], 2)) for r in lrows]
    body = (f'<p class="lead">Real source code (CPython\'s standard library), cut to exact token counts, each with a unique first line so the cache cannot help the cold run. '
            f'The same prompt is then sent again to measure prefix-cache reuse. Cold prompt processing stays above <b>{n(min(p for p in cold_pf if p), 0)} tok/s</b> all the way to {cats[-1]}.</p>'
            f'<h3 id="c-prefill">Cold prefill: prompt tokens processed per second</h3>{line_chart(cats, [("Cold prefill", BLUE, cold_pf, True)], "tok/s", "c-prefill")}'
            f'<h3 id="c-ttft">Time to first token: cold read</h3>{column_chart(cats, cold_ttft, BLUE, "s", "c-ttft", label_fmt=lambda v: f"{v:.1f}s" if v >= 1 else f"{v:.2f}s")}'
            f'<h3 id="c-warm">Time to first token: same prompt again (prefix cache)</h3>{column_chart(cats, warm_ttft, ORANGE, "s", "c-warm", label_fmt=lambda v: f"{v:.2f}s")}'
            f'<h3 id="c-depth">Decode speed at depth</h3>{line_chart(cats, [("Decode", BLUE, depth_decode, True)], "tok/s", "c-depth", label_fmt=lambda v: n(v, 1))}'
            + table(["Context", "~Prompt tokens", "~Cold first token", "~Prefill tok/s", "~Cached first token", "~Speed-up", "~Decode tok/s", "~DSpark kept", "~J/token"], rows))
    sections.append(("long", "Long-context prompts", "1K to 120K tokens of source code, cold and cached", body))

# Recall.
if recall:
    sizes = sorted({r["target"] for r in recall["rows"]})
    depths = sorted({r["depth"] for r in recall["rows"]})
    cells = []
    for s in sizes:
        cells.append(f'<div class="rc-row"><span class="rc-size">{ktok(s)}</span>')
        for d in depths:
            r = next(x for x in recall["rows"] if x["target"] == s and x["depth"] == d)
            ok = r["correct"]
            cells.append(f'<div class="rc-cell {"ok" if ok else "miss"}" data-tip="{ktok(s)} at {d:.0%}: {"found" if ok else "missed"} · {r["prompt"]:,} tokens · first token {secs(r["ttft"])} · decode {n(r["decode_tps"], 1)} tok/s">'
                         f'<b>{"&#10003; Found" if ok else "&#10007; Missed"}</b><span>{secs(r["ttft"])} to read</span></div>')
        cells.append("</div>")
    head = '<div class="rc-row rc-head"><span></span>' + "".join(f"<span>Fact at {d:.0%}</span>" for d in depths) + "</div>"
    body = (f'<p class="lead">A one-line comment with a random code is hidden at 10%, 50% or 90% depth inside real source code, and the model is asked for it. '
            f'<b>{recall["correct"]} of {recall["total"]}</b> answered exactly, up to {ktok(sizes[-1])} tokens.</p><div class="recall">{head}{"".join(cells)}</div>')
    first = load("recall-run1")
    if first:
        body += (f'<p class="note">Test history, for transparency: a first version framed the fact as a secret "vault access code", and the model declined to repeat it, '
                 f'so it was changed to a harmless release build tag. That version then scored {first["correct"]} of {first["total"]}: its one miss was at 64K, where the '
                 f'code excerpt ended mid-file and the model read the question as part of the code ("no actual task or question was included"). '
                 f'The final run above adds an explicit end-of-files marker before the question.</p>')
    probes = [(p, "different source excerpt") for p in (load("recall-64k-probe") or [])] + \
             [(p, "boundary check") for p in (load("recall-64k-boundary") or [])] + \
             [(p, "repeat at the failing length") for p in (load("recall-64k-repeat") or [])]
    if probes:
        probes.sort(key=lambda x: x[0]["prompt"])
        prow = [(n(p["prompt"]), esc(why), f'<span class="tag {"ok" if p["correct"] else "miss"}">{"found" if p["correct"] else "missed"}</span>',
                 esc((p.get("text") or "")[:90].replace("\n", " "))) for p, why in probes]
        wrong = [p for p, _ in probes if not p["correct"]]
        body += (f'<h3>Finding: a narrow miss just past 64K</h3><p class="lead">Both misses were in the 64K case, so it was probed further with a fact at 90% depth. '
                 f'Prompts of 63K, 65.5K and 70K tokens answered every time, but prompts of about <b>65,620 tokens</b> missed in {len(wrong)} of '
                 f'{sum(1 for p, _ in probes if 65600 <= p["prompt"] <= 65650)} tries: the model describes the code and says no question was asked, as if it does not see '
                 f'the question at the very end. 96K and 124K are unaffected. This looks like an edge case in the serving stack at that length (prompts are processed in '
                 f'2,048-token chunks, and the last chunk there is only about 84 tokens), not a general long-context limit. The raw probes are in the results folder.</p>'
                 + table(["~Prompt tokens", "Probe", "Result", "Start of the answer"], prow))
    sections.append(("recall", "Context up to 128K", "Needle-in-a-haystack recall", body))

# DSpark.
if per_pos:
    groups = [("Short bursts", bruns), ("Long context", [r["cold"] for r in lrows]), ("128K recall", recall["rows"] if recall else []), ("Coding run", coding["turns"] if coding else [])]
    grows = []
    for name, rs in groups:
        rs = [r for r in rs if r.get("spec") and r["spec"]["drafts"]]
        if not rs:
            continue
        d, dt, a = sum(r["spec"]["drafts"] for r in rs), sum(r["spec"]["drafted"] for r in rs), sum(r["spec"]["accepted"] for r in rs)
        grows.append((esc(name), n(len(rs)), n(d), n(dt), n(a), f"{100 * a / dt:.1f}%", f"{1 + a / d:.2f}"))
    body = (f'<p class="lead">DSpark drafts 7 tokens ahead; the model verifies them in one pass and keeps the longest correct run. Across all single-request tests, '
            f'<b>{100 * acc_accepted / acc_drafted:.1f}%</b> of drafted tokens were kept, so each verify step produced <b>{1 + acc_accepted / acc_drafts:.2f}</b> tokens on average. '
            f'Counts come from vLLM\'s own speculative-decoding counters, read before and after every request.</p>'
            f'<h3 id="c-pos">Acceptance by draft position</h3>{column_chart([f"#{i + 1}" for i in range(7)], [p * 100 for p in per_pos], BLUE, "%", "c-pos", label_fmt=lambda v: f"{v:.0f}%")}'
            f'<p class="note">Share of verify steps where the draft was still correct at that position. The first drafted token is kept about {per_pos[0] * 100:.0f}% of the time.</p>'
            f'<h3 id="c-depthacc">Acceptance by context length</h3>{line_chart(cats, [("Acceptance", BLUE, [a * 100 for a in depth_acc], True)], "%", "c-depthacc", ymax=60, label_fmt=lambda v: f"{v:.0f}%")}'
            + table(["Test", "~Requests", "~Verify steps", "~Drafted", "~Accepted", "~Kept", "~Tokens / step"], grows))
    sections.append(("dspark", "DSpark speculative decoding", "Acceptance, measured from vLLM's counters", body))

# Coding.
if coding:
    t = coding["turns"]
    rows = [(str(x["turn"]), n(x["prompt"]), n(x["cached"]), secs(x["ttft"]), n(x["completion"]), n(x["decode_tps"], 1),
             n(x["spec"]["acceptance"] * 100 if x["spec"]["acceptance"] is not None else None, 0) + "%",
             esc(", ".join(x["tools_called"]) or "(text)") + (f' <span class="tag {"ok" if x["tests"]["ok"] else "miss"}">{"no tests found" if not x["tests"]["ran"] else str(x["tests"]["ran"]) + (" tests pass" if x["tests"]["ok"] else " tests, some fail")}</span>' if x.get("tests") else ""))
            for x in t]
    s = coding["summary"]
    body = (f'<p class="lead">An agent-style session: system prompt, three tools (write a file, read a file, run the tests) and a task to build a small Python '
            f'package with unit tests. The harness really runs the model\'s tests (offline, in a throwaway container) and feeds the output back until they pass. '
            f'Result: <b>{"all tests pass" if coding["passed"] else "tests still failing"}</b>, {coding["tests_ran"]} tests, {len(coding["files"])} files, '
            f'{coding["lines"]} lines, {len(t)} turns in {coding["wall_s"]:.0f} seconds. Median decode <b>{n(s["decode_tps"]["median"], 1)} tok/s</b>; '
            f'the growing history is served from the prefix cache, so each turn starts in about <b>{secs(med([x["ttft"] for x in t[1:]]))}</b>.</p>'
            f'<h3 id="c-turns">Decode speed per turn</h3>{column_chart([str(x["turn"]) for x in t], [x["decode_tps"] for x in t], BLUE, "tok/s", "c-turns", label_fmt=lambda v: n(v, 0))}'
            + table(["Turn", "~Prompt", "~Cached", "~First token", "~Out", "~Decode tok/s", "~DSpark kept", "Tools"], rows)
            + f'<details><summary>Files written and final test output</summary><p>{esc(", ".join(coding["files"]))}</p><pre>{esc(coding["final_tests"])}</pre></details>')
    sections.append(("coding", "A normal coding run", "Multi-turn tool use with real test execution", body))

# Concurrency.
if crow:
    labels = [f'{r["n"]} × {r["label"]}' for r in crow]
    rows = [(esc(r["label"]), str(r["n"]), n(r["aggregate_tps"], 1), n(med(r["per_stream_decode"]), 1), secs(med(r["ttft"])), secs(max(r["ttft"])),
             n(r["spec"]["acceptance"] * 100 if r["spec"]["acceptance"] else None, 0) + "%", n(r["j_per_tok"], 2), n(r["avg_w"], 0)) for r in crow]
    body = (f'<p class="lead">Several chats at the same time (the server allows up to 8). Each asks for at least 400 tokens. Aggregate output rises to '
            f'<b>{n(peak_aggr, 0)} tok/s</b> while each stream slows down; energy per token falls as the GPU is shared.</p>'
            f'<h3 id="c-aggr">Aggregate output tok/s</h3>{column_chart(labels, [r["aggregate_tps"] for r in crow], BLUE, "tok/s", "c-aggr", label_fmt=lambda v: n(v, 0))}'
            f'<h3 id="c-per">Median decode per stream</h3>{column_chart(labels, [med(r["per_stream_decode"]) for r in crow], ORANGE, "tok/s", "c-per", label_fmt=lambda v: n(v, 0))}'
            + table(["Prompt", "~Streams", "~Aggregate tok/s", "~Per stream tok/s", "~Median first token", "~Slowest first token", "~DSpark kept", "~J/token", "~Avg W"], rows))
    sections.append(("concurrency", "Concurrent requests", "1, 2, 4 and 8 streams", body))

# Method.
sections.append(("method", "How it was measured", "", f"""<ul class="method">
<li>All requests went straight to vLLM's OpenAI API on the B70 (no proxy), streamed, with token usage returned at the end.</li>
<li><b>First token</b> is the first streamed token of any kind, including the model's reasoning. <b>First answer</b> is the first token of the visible reply.</li>
<li><b>Prefill</b> = uncached prompt tokens / time to first token. <b>Decode</b> = (output tokens - 1) / (end - first token), thinking tokens included.</li>
<li><b>DSpark</b> figures are deltas of vLLM's <code>spec_decode</code> counters around each request. Kept = accepted / drafted; tokens per step = 1 + accepted / drafts.</li>
<li><b>Energy</b> is the card's own energy counter (hwmon, whole board) over each request, so it includes idle power while the request runs.</li>
<li>Every prompt size was run once before measuring, so first-use kernel compilation does not appear in the results. Long prompts carry a unique first line, so cold runs never hit the prefix cache.</li>
<li>Sampling: temperature 0.6 (0.3 for the coding run), top-p 0.95. Output varies a little run to run, as with any sampled model.</li>
<li>vLLM's request counter was read around every phase and matched the harness exactly (bursts 24, long context 20, recall 12, coding 8, concurrency 19), so no other traffic reached the model during the measurements.</li>
<li>Concurrency was measured with the server allowing 8 requests at once (<code>--max-num-seqs 8</code>), after one unmeasured pass of the same tests; every other test ran with 1.</li>
</ul>"""))

# ---- page ---------------------------------------------------------------------------------------------------------
nav = "".join(f'<a href="#{sid}">{esc(title)}</a>' for sid, title, _, _ in sections)
blocks = "".join(f'<section id="{sid}"><header><h2>{esc(title)}</h2>{f"<p>{esc(sub)}</p>" if sub else ""}</header>{body}</section>' for sid, title, sub, body in sections)
date = env["date"].split()[0] if env else ""
PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inkling 2 Mini on Intel Arc Pro B70 · Benchmark</title>
<meta name="description" content="Detailed single-GPU benchmark of Inkling 2 Mini on an Intel Arc Pro B70 with vLLM XPU and DSpark: decode, prefill, first-token latency, 128K context, coding and concurrency.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {{ color-scheme: light; --blue:{BLUE}; --blue-deep:#00285A; --cyan:#00C7FD; --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --grid:{GRID}; --surface:#FFFFFF; --page:#F4F7FB; --good:{GOOD}; --bad:{BAD}; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:16px/1.6 Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }}
a {{ color:var(--blue); }}
.hero {{ background:linear-gradient(120deg, #001A3D 0%, var(--blue-deep) 38%, var(--blue) 100%); color:#fff; padding:56px 24px 40px; position:relative; overflow:hidden; }}
.hero::after {{ content:""; position:absolute; right:-120px; top:-140px; width:520px; height:520px; border-radius:50%; border:70px solid rgba(0,199,253,.14); }}
.wrap {{ max-width:1120px; margin:0 auto; position:relative; z-index:1; }}
.kicker {{ font:700 13px/1 Inter, sans-serif; letter-spacing:.14em; text-transform:uppercase; color:var(--cyan); margin:0 0 14px; }}
.hero h1 {{ font-size:clamp(30px, 5vw, 52px); line-height:1.08; margin:0 0 14px; font-weight:800; letter-spacing:-.02em; }}
.hero .sub {{ font-size:18px; color:#D6E6F7; max-width:760px; margin:0 0 30px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:12px; }}
.stat {{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18); border-radius:12px; padding:16px 18px; display:flex; flex-direction:column; gap:4px; }}
.stat-label {{ font-size:13px; color:#BFD6EE; font-weight:600; }}
.stat-value {{ font:700 32px/1.1 "JetBrains Mono", ui-monospace, monospace; color:#fff; }}
.stat-value small {{ font:600 14px Inter, sans-serif; color:#BFD6EE; margin-left:6px; }}
.stat-sub {{ font-size:12.5px; color:#BFD6EE; line-height:1.4; }}
nav.toc {{ position:sticky; top:0; z-index:5; background:rgba(255,255,255,.94); backdrop-filter:blur(8px); border-bottom:1px solid var(--grid); }}
nav.toc .wrap {{ display:flex; gap:4px; overflow-x:auto; padding:0 16px; }}
nav.toc a {{ white-space:nowrap; text-decoration:none; color:var(--ink2); font-weight:600; font-size:14px; padding:14px 12px; min-height:44px; display:inline-flex; align-items:center; border-bottom:3px solid transparent; }}
nav.toc a:hover {{ color:var(--blue); border-bottom-color:var(--blue); }}
main {{ padding:8px 24px 60px; }}
section {{ background:var(--surface); border:1px solid var(--grid); border-radius:14px; padding:28px 30px; margin-top:24px; scroll-margin-top:70px; }}
section > header {{ border-left:4px solid var(--blue); padding-left:14px; margin-bottom:16px; }}
section h2 {{ margin:0; font-size:24px; font-weight:800; letter-spacing:-.01em; }}
section > header p {{ margin:2px 0 0; color:var(--muted); font-size:14px; }}
h3 {{ font-size:16px; margin:26px 0 6px; color:var(--ink); }}
.lead {{ color:var(--ink2); max-width:900px; }}
.note {{ color:var(--muted); font-size:13.5px; margin:4px 0 0; }}
.chart {{ width:100%; height:auto; display:block; font-family:Inter, sans-serif; }}
.chart .grid {{ stroke:var(--grid); stroke-width:1; }}
.chart .tick {{ fill:var(--muted); font-size:12px; }}
.chart .tick.x {{ fill:var(--ink2); font-weight:600; font-size:12.5px; }}
.chart .vlabel {{ fill:var(--ink); font:700 12.5px "JetBrains Mono", monospace; }}
.chart .hit {{ cursor:pointer; }}
.legend {{ display:flex; gap:18px; font-size:13.5px; color:var(--ink2); margin:4px 0 6px; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:7px; vertical-align:-1px; }}
.tablewrap {{ overflow-x:auto; margin-top:18px; border:1px solid var(--grid); border-radius:10px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ background:#EEF3F9; color:var(--ink2); text-align:left; font-weight:700; font-size:12.5px; padding:10px 12px; white-space:nowrap; }}
td {{ padding:9px 12px; border-top:1px solid var(--grid); white-space:nowrap; }}
td.r, th.r {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.r {{ font-family:"JetBrains Mono", ui-monospace, monospace; font-size:13px; }}
table.kv td {{ white-space:normal; }}
table.kv td:first-child {{ width:210px; }}
.tag {{ display:inline-block; font-size:12px; font-weight:700; border-radius:999px; padding:1px 8px; margin-left:6px; }}
.tag.ok {{ background:#E6F4EA; color:var(--good); }} .tag.miss {{ background:#FCE8E6; color:var(--bad); }}
.recall {{ display:grid; gap:8px; margin-top:14px; }}
.rc-row {{ display:grid; grid-template-columns:70px repeat(3, minmax(0,1fr)); gap:8px; align-items:stretch; }}
.rc-head span {{ font-size:13px; font-weight:700; color:var(--ink2); text-align:center; }}
.rc-size {{ font:700 15px "JetBrains Mono", monospace; display:flex; align-items:center; }}
.rc-cell {{ border-radius:10px; padding:12px; text-align:center; display:flex; flex-direction:column; gap:2px; cursor:default; }}
.rc-cell b {{ font-size:15px; }} .rc-cell span {{ font-size:12.5px; color:var(--ink2); }}
.rc-cell.ok {{ background:#E6F4EA; border:1px solid #B7DFC1; }} .rc-cell.ok b {{ color:var(--good); }}
.rc-cell.miss {{ background:#FCE8E6; border:1px solid #F4C2BD; }} .rc-cell.miss b {{ color:var(--bad); }}
details {{ margin-top:16px; }} summary {{ cursor:pointer; font-weight:600; color:var(--blue); min-height:44px; display:flex; align-items:center; }}
pre {{ background:#0B1F3A; color:#DCE8F7; padding:14px; border-radius:10px; overflow-x:auto; font-size:12.5px; }}
code {{ font-family:"JetBrains Mono", monospace; font-size:.92em; background:#EEF3F9; padding:1px 5px; border-radius:4px; }}
ul.method li {{ margin:6px 0; color:var(--ink2); }}
footer {{ text-align:center; color:var(--muted); font-size:13px; padding:10px 24px 40px; }}
#tip {{ position:fixed; z-index:20; pointer-events:none; background:#0B1F3A; color:#fff; font-size:13px; padding:7px 10px; border-radius:8px; box-shadow:0 8px 24px rgba(0,0,0,.25); max-width:320px; opacity:0; transition:opacity .12s; }}
:focus-visible {{ outline:3px solid var(--cyan); outline-offset:2px; }}
@media (max-width:640px) {{ section {{ padding:20px 16px; }} main {{ padding:4px 12px 40px; }} .hero {{ padding:40px 16px 30px; }} .stat-value {{ font-size:26px; }} .rc-row {{ grid-template-columns:52px repeat(3, minmax(0,1fr)); }} }}
@media print {{ nav.toc {{ display:none; }} section {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
<p class="kicker">Independent benchmark · {esc(date)}</p>
<h1>Inkling 2 Mini on one Intel Arc Pro B70</h1>
<p class="sub">Thinking Machines' 26-billion-parameter looped MoE, served by vLLM on a single 32 GB Battlemage card with DSpark speculative decoding: short prompts, long context to 128K, a real coding session and concurrent load.</p>
<div class="stats">{"".join(hero_stats)}</div>
</div></header>
<nav class="toc" aria-label="Sections"><div class="wrap">{nav}</div></nav>
<main><div class="wrap">{blocks}</div></main>
<footer>Independent community benchmark. Not produced or endorsed by Intel or Thinking Machines. Intel and Arc are trademarks of Intel Corporation.</footer>
<div id="tip" role="tooltip"></div>
<script>
// Hover details for every chart mark and recall cell.
const tip = document.getElementById('tip');
document.addEventListener('pointermove', e => {{
  const t = e.target.closest('[data-tip]');
  if (!t) {{ tip.style.opacity = 0; return; }}
  tip.textContent = t.dataset.tip;
  tip.style.opacity = 1;
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + 'px'; tip.style.top = (e.clientY + 16) + 'px';
}});
</script>
</body>
</html>
"""
OUT.write_text(PAGE, encoding="utf-8")
print(f"wrote {OUT} ({len(PAGE) // 1024} KB), sections: {[s[0] for s in sections]}")
