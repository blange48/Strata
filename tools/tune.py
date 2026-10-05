#!/usr/bin/env python3
"""Tune a server config on this machine, end to end: the candidates come from tools/autoconfig.py's rules plus the
knobs worth measuring (PCIe share of the missed experts, resident K/V, pipeline groups); each one is served through
serve/server.py and measured over HTTP (total tok/s at several loads, time to first token on a long fresh prompt,
free RAM, greedy answers to compare); the best one is checked for exactness (tools/batch_test.py, and
tools/parking_test.py when parking is on) and written out with a report.

  python3 tools/tune.py --config strata-<model>.json                     # rules + default candidates
  python3 tools/tune.py --config strata-<model>.json --quick             # rules + PCIe share only
  python3 tools/tune.py --config strata-<model>.json --candidates rules,pcie0.6,kvres-half

The rates depend on the text: the experts it routes to and how many MTP drafts are accepted (English code questions
ran ~13 % faster than French prose on the same config).  Compare candidates with the same prompts - and pass your
own (--prompts, a JSON list of strings) to tune for what your users send.

It needs the GPUs to itself (stop any running server first) and takes several minutes per candidate.  The config
given is never changed: the winner goes to --out (default: <config>.tuned.json) and the report to <out>.md.  A
candidate only replaces the rules' config when it beats it by more than --margin (run-to-run noise).
"""
import argparse, json, os, secrets, socket, statistics, subprocess, sys, tempfile, threading, time
import urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import autoconfig as AC  # noqa: E402

LOAD_PROMPTS = [
    "Explain in detail how a B-tree works and why databases use it.",
    "Write a Python function that merges overlapping intervals, with tests, then explain it.",
    "Compare TCP and QUIC: connection setup, congestion control, multiplexing. Be precise.",
    "Tell the history of computing pi, from Archimedes to modern algorithms.",
]
QUALITY_PROMPTS = [
    "Explain in six sentences how a B+ tree works in a database.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List eight Linux commands useful to diagnose a network problem, one sentence each.",
]


# ------------------------------------------------------------------ candidates
def set_arg(args, name, value=None):
    a = list(args)
    if name in a:
        i = a.index(name)
        if value is None:
            return a
        a[i + 1] = str(value)
    else:
        a += [name] if value is None else [name, str(value)]
    return a


def drop_arg(args, name, n=1):
    a = list(args)
    if name in a:
        i = a.index(name)
        del a[i:i + 1 + n]
    return a


def candidates(base, wanted):
    """name -> config, all derived from `base` (the rules' config)."""
    out = {"rules": base}
    args = base["args"]
    kvres = int(AC.arg(args, "--kv-resident", 0) or 0)
    for name in wanted:
        c = json.loads(json.dumps(base))
        if name.startswith("pcie"):                         # pcie0.6: that share of the missed experts over PCIe
            c["args"] = set_arg(args, "--pcie-frac", name[4:])
        elif name == "kvres-half" and kvres > 1:
            c["args"] = set_arg(args, "--kv-resident", kvres // 2)
        elif name == "no-groups" and "--batch-groups" in args:
            c["args"] = drop_arg(args, "--batch-groups")
        elif name == "rules":
            continue
        else:
            print(f"tune: candidate {name!r} does not apply here, skipped", flush=True)
            continue
        out[name] = c
    return out


# ------------------------------------------------------------------ one served candidate
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def mem_available_gib():
    return AC.meminfo()[1] / AC.GIB


def chat(port, key, prompt, max_tokens, temperature, timeout=900):
    body = json.dumps({"model": "m", "max_tokens": max_tokens, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t


def load_level(port, key, clients, requests, max_tokens):
    """Total generated tok/s with `clients` concurrent clients, each sending `requests` requests (T=0.7)."""
    done, errs, lock = [], [], threading.Lock()

    def worker(i):
        for k in range(requests):
            try:
                d, _ = chat(port, key, LOAD_PROMPTS[(i + k) % len(LOAD_PROMPTS)], max_tokens, 0.7)
                with lock:
                    done.append(d["usage"]["completion_tokens"])
            except Exception as e:  # noqa: BLE001
                with lock:
                    errs.append(repr(e)[:120])

    t = time.time()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(clients)]
    [x.start() for x in th]
    [x.join() for x in th]
    return sum(done) / max(time.time() - t, 1e-9), errs


def serve_and_measure(name, cfg, a, workdir):
    path = workdir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=1))
    port, key = free_port(), secrets.token_hex(16)
    log = open(workdir / f"{name}.server.log", "w")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config", str(path),
                             "--port", str(port), "--api-key", key], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    res = {"name": name, "args": cfg["args"], "ok": False}
    try:
        t0 = time.time()
        while time.time() - t0 < a.start_timeout:          # ready = a first token comes back
            if proc.poll() is not None:
                res["error"] = "the server ended while starting (see the server log)"
                return res
            try:
                chat(port, key, "Hi", 4, 0, timeout=30)
                break
            except (urllib.error.URLError, OSError, ValueError):
                time.sleep(10)
        else:
            res["error"] = f"not ready after {a.start_timeout} s"
            return res
        res["start_s"] = round(time.time() - t0)
        res["ram_free_gib"] = round(mem_available_gib(), 1)
        res["rates"] = {}
        for c in a.clients:
            rate, errs = load_level(port, key, c, a.requests, a.max_tokens)
            res["rates"][c] = round(rate, 1)
            if errs:
                res.setdefault("errors", []).extend(errs[:2])
            print(f"   {name}: {c} client(s) {rate:6.1f} tok/s{'  errors: ' + str(len(errs)) if errs else ''}", flush=True)
        if a.long_prompt:
            words = "the server reads a long technical document with code and tables".split()
            text = f"[{secrets.token_hex(4)}] " + " ".join(words[i % len(words)] for i in range(int(a.long_prompt * 0.75)))
            _, dt = chat(port, key, text + "\nSummarize in one word.", 1, 0)
            res["long_prompt_s"] = round(dt, 2)
            print(f"   {name}: {a.long_prompt}-token prompt {dt:.2f} s", flush=True)
        res["answers"] = [chat(port, key, p, 160, 0)[0]["choices"][0]["message"]["content"] for p in QUALITY_PROMPTS]
        res["ok"] = True
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


# ------------------------------------------------------------------ exactness of the winner
def exact(cfg, workdir):
    """tools/batch_test.py (greedy slots equal to solo) and tools/parking_test.py when parking is on."""
    path = workdir / "exact.json"
    path.write_text(json.dumps(cfg, indent=1))
    batch = int(cfg.get("parallel") or AC.arg(cfg["args"], "--batch", 0) or 0)
    extra = "--adapt-every 1000000 --pcie-frac 0"
    report = []
    if batch > 1:
        for n in sorted({batch, max(2, batch // 2 - 1)}):
            r = subprocess.run([sys.executable, str(HERE / "batch_test.py"), "--exe", cfg["exe"], "--config", str(path),
                                "--batch", str(batch), "--n", str(n), "--max-new", "120", "--extra", extra],
                               capture_output=True, text=True)
            same = r.stdout.count("IDENTICAL")
            report.append(f"batch_test {n} slots: {same}/{n} identical to solo")
            if same != n:
                return False, report
    if int(AC.arg(cfg["args"], "--conversation-cache-mib", 0) or 0) > 0:
        r = subprocess.run([sys.executable, str(HERE / "parking_test.py"), "--exe", cfg["exe"], "--config", str(path),
                            "--extra", extra], capture_output=True, text=True)
        line = next((x for x in r.stdout.splitlines() if x.startswith("follow-up tokens")), "no result")
        report.append(f"parking_test: {line}")
        if "IDENTICAL" not in line:
            return False, report
    return True, report


# ------------------------------------------------------------------ choice and report
def score(res, best):
    return statistics.mean(res["rates"][c] / best[c] for c in best) if res.get("ok") else -1


def first_difference(a, b):
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    return None if k is None and len(a) == len(b) else (k if k is not None else min(len(a), len(b)))


def report_md(results, ranking, winner, exact_lines, cards, ram_total, path_cfg):
    clients = sorted({c for r in results.values() if r.get("ok") for c in r["rates"]})
    base = results.get("rules", {})
    lines = ["# Strata tuning report", "",
             "GPUs: " + ", ".join(f"{g['index']}: {g['name']} {g['mib'] / 1024:.0f} GiB PCIe x{g['width']}" for g in cards),
             f"RAM: {ram_total / AC.GIB:.0f} GiB", "",
             "| Candidate | " + " | ".join(f"{c} cl. tok/s" for c in clients) + " | long prompt | free RAM | answers = rules |",
             "|---|" + "---:|" * (len(clients) + 3)]
    for name, _ in ranking:
        r = results[name]
        if not r.get("ok"):
            lines.append(f"| {name} | {r.get('error', 'failed')} |")
            continue
        same = sum(first_difference(x, y) is None for x, y in zip(base.get("answers", []), r["answers"])) if base.get("ok") else "-"
        lines.append(f"| {'**' + name + '**' if name == winner else name} | "
                     + " | ".join(str(r["rates"].get(c, "-")) for c in clients)
                     + f" | {r.get('long_prompt_s', '-')} s | {r['ram_free_gib']} GiB | {same}/{len(QUALITY_PROMPTS)} |")
    lines += ["", f"Chosen: **{winner}**, written to `{path_cfg}`.", "", "Exactness of the chosen config:"]
    lines += [f"- {x}" for x in exact_lines] or ["- not checked"]
    lines += ["", "Answers that differ from the rules' config are expected when the PCIe share or the resident K/V",
              "changes (other rounding on the CPU and the GPU); they are greedy runs, not a quality score."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="the tuned config (default <config>.tuned.json); the report goes next to it (.md)")
    ap.add_argument("--candidates", default="rules,pcie0.6,pcie1.0,kvres-half,no-groups")
    ap.add_argument("--quick", action="store_true", help="only the rules and two PCIe shares")
    ap.add_argument("--no-rules", action="store_true", help="tune the given config as it is, without autoconfig's rules")
    ap.add_argument("--clients", default="1,4,8")
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--long-prompt", type=int, default=45000, help="tokens of the fresh prompt timed (0: skip)")
    ap.add_argument("--margin", type=float, default=0.03, help="a candidate must beat the rules by this much")
    ap.add_argument("--start-timeout", type=int, default=1500)
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--prompts", help="a JSON list of prompts for the load test (default: four English questions)")
    a = ap.parse_args()
    a.clients = [int(x) for x in a.clients.split(",")]
    if a.prompts:
        LOAD_PROMPTS[:] = [str(x) for x in json.loads(Path(a.prompts).read_text())]
    cfg_path = Path(a.config)
    cfg = json.loads(cfg_path.read_text())
    cards = [g for g in AC.gpus() if g["index"] in (cfg.get("gpu") or [g["index"] for g in AC.gpus()])]
    if not cards:
        raise SystemExit("tune: nvidia-smi found no GPU")
    ram_total, _ = AC.meminfo()
    native = AC.arg(cfg["args"], "--native")
    layers = (AC.gguf_block_count(native) if native else None) or 48
    if a.no_rules:
        base = cfg
    else:
        s, notes = AC.plan(cfg, cards, ram_total, layers)
        base = AC.apply(cfg, s)
        for n in notes:
            print("note:", n, flush=True)
    wanted = ["rules", "pcie0.6", "pcie1.0"] if a.quick else [x.strip() for x in a.candidates.split(",") if x.strip()]
    cands = candidates(base, wanted)
    workdir = Path(tempfile.mkdtemp(prefix="strata-tune-"))
    print(f"tune: {len(cands)} candidates ({', '.join(cands)}), logs in {workdir}", flush=True)
    results = {}
    for name, c in cands.items():
        print(f"-- {name}", flush=True)
        results[name] = serve_and_measure(name, c, a, workdir)
        if not results[name]["ok"]:
            print(f"   {name}: {results[name].get('error')}", flush=True)
    ok = {n: r for n, r in results.items() if r.get("ok")}
    if not ok:
        raise SystemExit(f"tune: no candidate could be served (logs in {workdir})")
    best = {c: max(r["rates"][c] for r in ok.values()) for c in a.clients}
    ranking = sorted(((n, score(results[n], best)) for n in results), key=lambda x: -x[1])
    rules_score = score(results["rules"], best) if "rules" in ok else -1
    order = [n for n, s in ranking if n in ok and (n == "rules" or s > rules_score * (1 + a.margin))]
    order += [n for n, _ in ranking if n in ok and n not in order]
    winner, exact_lines = None, []
    for n in order:
        if a.skip_exact:
            winner = n
            break
        print(f"-- exactness of {n}", flush=True)
        good, lines = exact(cands[n], workdir)
        exact_lines = lines
        for x in lines:
            print("  ", x, flush=True)
        if good:
            winner = n
            break
    if winner is None:
        raise SystemExit("tune: no candidate passed the exactness checks; nothing written")
    out = Path(a.out) if a.out else cfg_path.with_name(cfg_path.name.replace(".json", "") + ".tuned.json")
    out.write_text(json.dumps(cands[winner], indent=1) + "\n")
    md = out.with_suffix(".md")
    md.write_text(report_md(results, ranking, winner, exact_lines, cards, ram_total, out))
    print(f"tune: chose {winner}; config {out}, report {md}", flush=True)
    print(f"tune: server logs kept in {workdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
