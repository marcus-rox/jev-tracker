"""Build docs/src/jev_alternatives_body.html from the crawler's candidates, triage verdicts and queue.

Run from the repo root:  python3 docs/src/build_jev_alternatives.py
Then compile the fragment into docs/JEV_ALTERNATIVES.html with the artifact kit.
"""
from __future__ import annotations

import collections
import glob
import html
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/src/jev_alternatives_body.html"

BENCHED = {
    "matilda": r"maincode/matilda-jev",
    "autotrust": r"autotrust/jev-27b(@|$)",
    "clef": r"cloudflare/clef(-flash)?(@|$)",
    "jevany": r"jevany-qwen3\.5-4b",
    "rsi_jev": r"rsi-jev-v3",
    "minicpm5_jev": r"minicpm5-2b-jev",
    "d1": r"liquid.*\bd1\b|/d1\b",
}

BUCKETS = [
    ("runnable", "Runnable with today's harness", "Nothing: Kev / Laya family or a hosted API with a provisioned key."),
    ("quant", "Quantised / exported weights (GGUF, MLX, ONNX)", "A llama.cpp / MLX / ONNX serving source, or find the bf16 original."),
    ("arch", "Own architecture / serving", "A harness source that loads the model and reads P(true) or the ordinal readout."),
    ("unofficial", "Unofficial checkpoint of a supported family", "Allow-list the repo in the kev / laya source (low effort, low expected value)."),
    ("vision", "Vision / multimodal input", "A new request format carrying images; out of scope for the text benchmark."),
    ("noweights", "No weights / code released yet", "Blocked on the author."),
    ("api", "Hosted API: key or URL not provisioned", "A Modal Secret with the key, or a hosted URL for the self-hosted server."),
    ("kevbase", "Kev recipe on an unverified base model", "Verify the kev source on that base (Gemma-4, MiniCPM) then run."),
    ("budget", "Over GPU budget", "Approve the budget or a smaller GPU."),
    ("other", "Other", "See the triage reason."),
]
BUCKET_TITLE = {k: t for k, t, _ in BUCKETS}


def bucket(verdict: str, reason: str) -> str:
    if verdict == "runnable":
        return "runnable"
    r = reason.lower()
    if re.search(r"vision|multimodal|image", r):
        return "vision"
    if re.search(r"gguf|onnx|mlx|awq|gptq|quant|\bexport", r):
        return "quant"
    if re.search(r"hosted url|api source needs|modal secret|needs an? .*key|key not", r):
        return "api"
    if "budget" in r:
        return "budget"
    if re.search(r"no released|no weights|not released|weights.*not|without weights|code release|no checkpoint|no public", r):
        return "noweights"
    if re.search(r"unofficial|only admits|official repos", r):
        return "unofficial"
    if re.search(r"only verified on|recipe on", r):
        return "kevbase"
    if re.search(r"own architecture|own serving|serving stack|request format|no harness source|own .*package|engine|wrapper|readout|server", r):
        return "arch"
    return "other"


def model_of(key: str) -> str:
    m = re.match(r"(?:hf:|https://huggingface\.co/)([^/@]+/[^/@]+)", key)
    if not m:
        return ""
    org, name = m.group(1).split("/")
    stem = re.sub(
        r"[-_.]?(gguf|mlx|onnx|awq|gptq|fp8|int[48]|q\d(_[a-z0-9]+)*|[0-9]+bit|bf16|f16|fp16|nf4|exl2|mlc|coreml|openvino|tflite|lora)(?=[-_.]|$)",
        "",
        name,
        flags=re.I,
    ).strip("-_.")
    return f"{org}/{stem}"


def load():
    cands = {}
    for f in sorted(glob.glob(str(ROOT / "crawler/candidates/*.jsonl"))):
        for line in open(f):
            if line.strip():
                d = json.loads(line)
                cands.setdefault(d["key"], d)
    verdicts = {}
    for f in sorted(glob.glob(str(ROOT / "crawler/triage/*.yaml"))):
        for d in yaml.safe_load(open(f)) or []:
            verdicts.setdefault(d["key"], d)
    queue = {it["url"]: it for it in json.load(open(ROOT / "data/queue.json"))["items"]}
    return cands, verdicts, queue


def main() -> None:
    cands, verdicts, queue = load()
    rows = []
    not_jev = collections.Counter()
    for key, v in verdicts.items():
        c = cands[key]
        if v["verdict"] == "not_jev":
            not_jev[v["reason"].split(";")[0].split(":")[0].strip().lower()[:60]] += 1
            continue
        fam = next((f for f, pat in BENCHED.items() if re.search(pat, key.lower())), "")
        q = queue.get(c["url"])
        if q:
            status = f"queue: {q['status']}"
        elif v["verdict"] == "runnable":
            status = "ran today" if "run today" in v["reason"] else "benchmarked"
        elif fam:
            status = "benchmarked"
        else:
            status = "untouched"
        rows.append(
            dict(
                key=key, url=c["url"], title=c["title"], snippet=c["snippet"], source=c["source"],
                first_seen=c["first_seen"][:10], verdict=v["verdict"], reason=v["reason"],
                bucket=bucket(v["verdict"], v["reason"]), status=status, config=v["config"] or "",
                model=model_of(key),
            )
        )

    groups: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        groups[r["model"] or r["url"]].append(r)
    n_hits = len(verdicts)
    n_alt = len(rows)
    n_models = sum(1 for g in groups if "/" in g and not g.startswith("http"))
    n_untouched_models = sum(1 for g, rs in groups.items() if "/" in g and not g.startswith("http") and all(r["status"] == "untouched" for r in rs))
    by_bucket = collections.Counter(r["bucket"] for r in rows)
    models_by_bucket = collections.Counter()
    for g, rs in groups.items():
        if "/" in g and not g.startswith("http"):
            models_by_bucket[collections.Counter(r["bucket"] for r in rs).most_common(1)[0][0]] += 1
    by_status = collections.Counter(r["status"] for r in rows)
    sources = collections.Counter(r["source"] for r in rows)
    queue_counts = collections.Counter(it["status"] for it in queue.values())

    e = html.escape

    def opt(values, labels=None):
        labels = labels or {}
        return "".join(f'<option value="{e(v)}">{e(labels.get(v, v))}</option>' for v in values)

    # ---- model roll-up (one row per Hugging Face model stem, or per non-HF hit)
    def group_row(g, rs):
        rs = sorted(rs, key=lambda r: r["first_seen"])
        is_model = "/" in g and not g.startswith("http")
        name = g if is_model else rs[0]["title"]
        link = f"https://huggingface.co/{g}" if is_model else rs[0]["url"]
        b = collections.Counter(r["bucket"] for r in rs).most_common(1)[0][0]
        st = sorted({r["status"] for r in rs}, key=lambda s: ("untouched" in s, s))[0]
        src = sorted({r["source"] for r in rs})
        reason = rs[0]["reason"]
        return (
            f'<tr data-bucket="{b}" data-status="{e(st)}" data-source="{e(src[0])}" data-text="{e((name + " " + reason + " " + " ".join(r["title"] for r in rs)).lower())}">'
            f'<th scope="row"><a href="{e(link)}">{e(name)}</a></th>'
            f'<td data-numeric>{len(rs)}</td>'
            f'<td>{e(BUCKET_TITLE[b])}</td>'
            f'<td>{e(st)}</td>'
            f'<td>{e(", ".join(src))}</td>'
            f'<td>{e(reason)}</td>'
            f'<td>{e(rs[0]["first_seen"])}</td>'
            "</tr>"
        )

    order = {k: i for i, (k, _, _) in enumerate(BUCKETS)}
    sorted_groups = sorted(groups.items(), key=lambda kv: (order[collections.Counter(r["bucket"] for r in kv[1]).most_common(1)[0][0]], -len(kv[1]), kv[0].lower()))
    model_rows = "\n".join(group_row(g, rs) for g, rs in sorted_groups)

    hit_rows = "\n".join(
        f'<tr data-bucket="{r["bucket"]}" data-status="{e(r["status"])}" data-source="{e(r["source"])}" data-text="{e((r["title"] + " " + r["reason"] + " " + r["snippet"]).lower())}">'
        f'<th scope="row"><a href="{e(r["url"])}">{e(r["title"])}</a></th>'
        f'<td>{e(r["source"])}</td><td>{e(BUCKET_TITLE[r["bucket"]])}</td><td>{e(r["status"])}</td>'
        f'<td>{e(r["reason"])}</td><td>{e(r["config"])}</td><td>{e(r["first_seen"])}</td></tr>'
        for r in sorted(rows, key=lambda r: (order[r["bucket"]], r["source"], r["title"].lower()))
    )

    bucket_table_rows = "\n".join(
        f'<tr><th scope="row">{e(t)}</th><td data-numeric>{by_bucket.get(k, 0)}</td><td data-numeric>{models_by_bucket.get(k, 0)}</td><td>{e(need)}</td></tr>'
        for k, t, need in BUCKETS if by_bucket.get(k, 0)
    )
    funnel_rows = "\n".join(
        f'<tr><th scope="row">{e(a)}</th><td data-numeric>{n}</td></tr>'
        for a, n in [
            ("Hits crawled (all sources)", n_hits),
            ("Triaged not_jev (people, CISA KEV, apps on Jev…)", n_hits - n_alt),
            ("Real Jev alternatives (runnable + needs_adapter)", n_alt),
            ("Distinct Hugging Face models among them", n_models),
            ("…never touched (no run, not in queue)", n_untouched_models),
            ("Benchmarked hits", by_status.get("benchmarked", 0) + by_status.get("ran today", 0)),
            ("In the approval queue (proposed)", queue_counts.get("proposed", 0)),
        ]
    )
    not_jev_rows = "\n".join(f"<tr><th scope=\"row\">{e(k)}</th><td data-numeric>{n}</td></tr>" for k, n in not_jev.most_common(10))
    source_rows = "\n".join(f"<tr><th scope=\"row\">{e(k)}</th><td data-numeric>{n}</td></tr>" for k, n in sources.most_common())

    statuses = sorted(by_status, key=lambda s: ("untouched" in s, s))
    filter_bar = lambda tid: f"""
<div class="a-toolbar" data-jev-filters="#{tid}">
  <label class="a-label">What is missing <select class="a-select" data-key="bucket"><option value="all">all</option>{opt([k for k, _, _ in BUCKETS if by_bucket.get(k)], BUCKET_TITLE)}</select></label>
  <label class="a-label">Status <select class="a-select" data-key="status"><option value="all">all</option>{opt(statuses)}</select></label>
  <label class="a-label">Source <select class="a-select" data-key="source"><option value="all">all</option>{opt([s for s, _ in sources.most_common()])}</select></label>
  <label class="a-label">Search <input class="a-select" type="search" data-key="text" placeholder="name, reason…"></label>
  <span class="a-label" data-count></span>
</div>"""

    body = f"""
<nav class="a-section" aria-labelledby="toc-heading">
  <h2 class="a-section__title" id="toc-heading">Contents</h2>
  <ul class="a-prose">
    <li><a href="#answer-heading">1. Why only {queue_counts.get('proposed', 0)} await approval</a></li>
    <li><a href="#funnel-heading">2. Funnel from crawl to queue (Figure 1, Table 1)</a></li>
    <li><a href="#models-heading">3. Every Jev alternative, one row per model (Table 2)</a></li>
    <li><a href="#hits-heading">4. Every raw hit (Table 3)</a></li>
    <li><a href="#method-heading">5. Method, sources and caveats</a></li>
  </ul>
</nav>

<section class="a-section" aria-labelledby="answer-heading">
  <h2 class="a-section__title" id="answer-heading">1. Why only {queue_counts.get('proposed', 0)} await approval</h2>
  <ul class="a-prose">
    <li>The crawler has seen <strong>{n_hits:,}</strong> hits. Triage marked <strong>{n_hits - n_alt:,}</strong> of them <code>not_jev</code> (people called Kev, CISA KEV, keV physics, apps built on Jev) — they are not alternatives and are not in this doc beyond Table 5.</li>
    <li><strong>{n_alt}</strong> hits are real Jev alternatives. Only <strong>{by_bucket['runnable']}</strong> are <code>runnable</code> with today's harness, and those are almost all posts about Kev and Laya, which are already benchmarked.</li>
    <li>The other <strong>{n_alt - by_bucket['runnable']}</strong> are <code>needs_adapter</code>: the harness cannot call them yet. The automation only promotes a <code>needs_adapter</code> hit to the approval queue when it judges the adapter "worth it" — that judgement is what left {queue_counts.get('proposed', 0)} in the queue, not the supply of models.</li>
    <li>Collapsing quantised variants, the {n_alt} hits are <strong>{n_models} distinct Hugging Face models</strong> plus {len(groups) - n_models} repos/posts without a Hugging Face link; <strong>{n_untouched_models}</strong> of those models have never been run or queued. That is the list to prioritise (Table 2).</li>
  </ul>
  <div class="a-grid">
    <div class="a-metric"><div class="a-metric__label">Hits crawled</div><div class="a-metric__value">{n_hits:,}</div></div>
    <div class="a-metric"><div class="a-metric__label">Real Jev alternatives</div><div class="a-metric__value">{n_alt}</div></div>
    <div class="a-metric"><div class="a-metric__label">Distinct HF models</div><div class="a-metric__value">{n_models}</div><div class="a-metric__delta">{n_untouched_models} never run or queued</div></div>
    <div class="a-metric"><div class="a-metric__label">Awaiting approval</div><div class="a-metric__value">{queue_counts.get('proposed', 0)}</div><div class="a-metric__delta">{queue_counts.get('queued', 0)} queued · {queue_counts.get('failed', 0)} failed</div></div>
  </div>
</section>

<section class="a-section" aria-labelledby="funnel-heading">
  <h2 class="a-section__title" id="funnel-heading">2. Funnel from crawl to queue</h2>
  <figure class="a-panel" data-a-chart="bar" data-a-chart-label="Hits at each stage from crawl to approval queue">
    <figcaption class="a-panel__title">Figure 1. Hits at each stage, crawl → triage → queue. Legend: one bar per stage; the value is a count of hits (or models where stated).</figcaption>
    <div class="a-table-scroll"><table class="a-table"><caption class="a-visually-hidden">Funnel counts</caption>
      <thead><tr><th scope="col">Stage</th><th scope="col" data-numeric>Count</th></tr></thead>
      <tbody>{funnel_rows}</tbody></table></div>
  </figure>
  <div class="a-table-scroll">
    <table class="a-table">
      <caption class="a-panel__title">Table 1. Why the alternatives are not runnable, and what each group needs. Columns: hits = raw crawler hits in the group; models = distinct Hugging Face models whose dominant verdict is this group.</caption>
      <thead><tr><th scope="col">What is missing</th><th scope="col" data-numeric>Hits</th><th scope="col" data-numeric>Models</th><th scope="col">What an adapter needs</th></tr></thead>
      <tbody>{bucket_table_rows}</tbody>
    </table>
  </div>
</section>

<section class="a-section" aria-labelledby="models-heading">
  <h2 class="a-section__title" id="models-heading">3. Every Jev alternative, one row per model</h2>
  <p class="a-section__note">Table 2. One row per distinct Hugging Face model (quantised / MLX / GGUF variants of the same weights collapsed into "variants") or per non-Hugging-Face repo/post. Sorted: runnable first, then by what is missing, then by number of variants. Status: benchmarked = results on the dashboard; ran today = the hourly run launched it; queue: proposed / queued / failed = in data/queue.json; untouched = nothing has happened yet. Filters combine; the search box matches name, reason and titles.</p>
  {filter_bar('models-table')}
  <div class="a-table-scroll">
    <table class="a-table" id="models-table" data-a-sticky-columns="1">
      <caption class="a-visually-hidden">Jev alternatives, one row per model</caption>
      <thead><tr><th scope="col">Model / repo</th><th scope="col" data-numeric>Variants</th><th scope="col">What is missing</th><th scope="col">Status</th><th scope="col">Sources</th><th scope="col">Triage reason</th><th scope="col">First seen</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table>
  </div>
  <p class="a-empty" hidden data-empty-for="#models-table">No model matches these filters.</p>
</section>

<section class="a-section" aria-labelledby="hits-heading">
  <h2 class="a-section__title" id="hits-heading">4. Every raw hit</h2>
  <details class="a-disclosure">
    <summary>Table 3. All {n_alt} raw hits triaged runnable or needs_adapter (one row per crawler hit; the same model can appear several times — once per export, post or pinned revision). Same filters as Table 2.</summary>
    {filter_bar('hits-table')}
    <div class="a-table-scroll">
      <table class="a-table" id="hits-table" data-a-sticky-columns="1">
        <caption class="a-visually-hidden">All raw hits</caption>
        <thead><tr><th scope="col">Hit</th><th scope="col">Source</th><th scope="col">What is missing</th><th scope="col">Status</th><th scope="col">Triage reason</th><th scope="col">Config</th><th scope="col">First seen</th></tr></thead>
        <tbody>{hit_rows}</tbody>
      </table>
    </div>
    <p class="a-empty" hidden data-empty-for="#hits-table">No hit matches these filters.</p>
  </details>
</section>

<section class="a-section" aria-labelledby="method-heading">
  <h2 class="a-section__title" id="method-heading">5. Method, sources and caveats</h2>
  <ul class="a-prose">
    <li>Hits: <code>crawler/candidates/*.jsonl</code> (deduplicated by <code>key</code> against <code>crawler/seen.jsonl</code>). Verdicts and reasons: <code>crawler/triage/*.yaml</code>, written by the hourly Devin run. Queue: <code>data/queue.json</code>.</li>
    <li>"What is missing" is a keyword grouping of the free-text triage reason; "Other" holds the reasons that fit no group. Read the reason column when it matters.</li>
    <li>"Distinct model" collapses Hugging Face repos whose name differs only by an export/quantisation suffix (gguf, mlx, 4bit, q4_k_m, awq, lora…). Different fine-tunes by the same author stay separate.</li>
    <li>"Benchmarked" is inferred from the triage reason or from a known family in <code>data/registry.yaml</code>; it does not say how the model scored — the dashboard does.</li>
    <li>No quality, cost or latency numbers here: nothing in this list has run unless its status says so.</li>
  </ul>
  <div class="a-grid a-grid--halves">
    <div class="a-table-scroll"><table class="a-table"><caption class="a-panel__title">Table 4. Alternatives by crawler source (hits).</caption>
      <thead><tr><th scope="col">Source</th><th scope="col" data-numeric>Hits</th></tr></thead><tbody>{source_rows}</tbody></table></div>
    <div class="a-table-scroll"><table class="a-table"><caption class="a-panel__title">Table 5. Top reasons for the {n_hits - n_alt:,} not_jev hits (excluded from this doc).</caption>
      <thead><tr><th scope="col">Reason</th><th scope="col" data-numeric>Hits</th></tr></thead><tbody>{not_jev_rows}</tbody></table></div>
  </div>
</section>

<script>
document.querySelectorAll('[data-jev-filters]').forEach(function (bar) {{
  var table = document.querySelector(bar.getAttribute('data-jev-filters'));
  var empty = document.querySelector('[data-empty-for="' + bar.getAttribute('data-jev-filters') + '"]');
  var count = bar.querySelector('[data-count]');
  var controls = bar.querySelectorAll('[data-key]');
  function apply() {{
    var shown = 0, total = 0;
    table.querySelectorAll('tbody tr').forEach(function (row) {{
      total += 1;
      var ok = true;
      controls.forEach(function (c) {{
        var k = c.getAttribute('data-key'), v = c.value;
        if (!v || v === 'all') return;
        var cell = row.getAttribute('data-' + k) || '';
        ok = ok && (k === 'text' ? cell.indexOf(v.toLowerCase()) !== -1 : cell === v);
      }});
      row.hidden = !ok;
      if (ok) shown += 1;
    }});
    if (empty) empty.hidden = shown !== 0;
    if (count) count.textContent = shown + ' of ' + total + ' rows';
  }}
  controls.forEach(function (c) {{ c.addEventListener('change', apply); c.addEventListener('input', apply); }});
  apply();
}});
</script>
"""
    OUT.write_text(body.strip() + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes): {n_hits} hits, {n_alt} alternatives, {n_models} models, {n_untouched_models} untouched")


if __name__ == "__main__":
    main()
