#!/usr/bin/env python3
"""Local web dashboard for a running NER annotation job (standard library only).

    python scripts/ner/monitor.py [--port 8765]      then open http://localhost:8765

Reads the newest progress_<signature>.json written by scripts/run_ner_annotate.py, the tail of the
newest run_*.log, and `nvidia-smi`, and refreshes every few seconds. It only reads files; stopping
it never affects the annotation job. Binds to localhost.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = REPO_ROOT / "data" / "processed" / "ner_dataset" / "annotations"


def newest(pattern: str) -> Path | None:
    files = sorted(ANN_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return dict(util=int(out[0]), mem_used=int(out[1]), mem_total=int(out[2]), temp=int(out[3]))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def status() -> dict:
    prog_path, log_path = newest("progress_*.json"), newest("run_*.log")
    prog = json.loads(prog_path.read_text()) if prog_path else None
    alive = None
    if prog:
        try:
            os.kill(prog["pid"], 0)
            alive = True
        except OSError:
            alive = False
    return {"progress": prog, "process_alive": alive, "heartbeat_age_s": time.time() - prog_path.stat().st_mtime if prog_path else None,
            "gpu": gpu(), "log_tail": log_path.read_text().splitlines()[-14:] if log_path else [], "now": time.time()}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NER Annotation Monitor</title><style>
:root{--bg:#fafafa;--fg:#1d2329;--mut:#66727d;--card:#fff;--line:#d9dee3;--acc:#2a78d6;--ok:#2e7d32;--bad:#c62828;--warn:#b26a00}
@media(prefers-color-scheme:dark){:root{--bg:#14181c;--fg:#e6e9ec;--mut:#97a3ad;--card:#1d2329;--line:#333c44;--acc:#6aa9f0;--ok:#6cc070;--bad:#ef6a6a;--warn:#e0a040}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}main{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0}.sub{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.tile b{display:block;font-size:22px}.tile span{color:var(--mut);font-size:12px}
#bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden}#bar div{height:100%;background:var(--acc);width:0}
.pill{display:inline-block;padding:1px 10px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}pre{margin:0;font-size:12px;white-space:pre-wrap;word-break:break-all;color:var(--mut)}
.row{display:flex;gap:8px;align-items:center;margin:3px 0;font-size:13px}.row i{display:block;height:10px;background:var(--acc);border-radius:3px}.row span:first-child{width:90px}svg{width:100%;height:60px}
</style></head><body><main><h1>NER Annotation Monitor <span id="pill" class="pill" style="background:var(--mut)">…</span></h1>
<div class="sub" id="sub"></div>
<div class="card"><div id="bar"><div></div></div><div class="sub" id="barlbl" style="margin-top:6px"></div></div>
<div class="card tiles" id="tiles"></div>
<div class="card"><b>Seconds per 256-utterance chunk</b><svg id="spark" viewBox="0 0 300 60" preserveAspectRatio="none"></svg><div class="sub" id="sparklbl"></div></div>
<div class="card"><b>Entities labeled so far</b><div id="labels"></div></div>
<div class="card"><b>Log</b><pre id="log"></pre></div></main>
<script>
const $=id=>document.getElementById(id), fmt=s=>{if(s==null)return"–";const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?h+"h "+m+"m":m+"m"};
async function tick(){let d;try{d=await (await fetch("/status")).json()}catch(e){$("pill").textContent="monitor offline";return}
const p=d.progress;if(!p){$("pill").textContent="no run found";return}
const stale=d.heartbeat_age_s>1500,dead=d.process_alive===false&&p.status==="running";
let st=p.status==="finished"?["finished","var(--ok)"]:p.status==="error"||dead?["stopped / error","var(--bad)"]:stale?["stalled?","var(--warn)"]:["running","var(--ok)"];
$("pill").textContent=st[0];$("pill").style.background=st[1];
$("sub").textContent=p.model+" · signature "+p.signature+" · last update "+Math.round(d.heartbeat_age_s)+"s ago (a chunk takes ~10 min)"+(dead?" · PROCESS NOT FOUND":"");
const pct=100*p.annotated/p.pool_size;$("bar").firstElementChild.style.width=pct+"%";
$("barlbl").textContent=p.annotated.toLocaleString()+" / "+p.pool_size.toLocaleString()+" utterances ("+pct.toFixed(1)+"%)";
const g=d.gpu,T=[["Remaining",fmt(p.eta_seconds)],["Rate",p.rate_utt_per_s?(1/p.rate_utt_per_s).toFixed(2)+" s/utt":"–"],["Done this run",p.this_run_done.toLocaleString()],
["Valid",(100*p.ok_rate).toFixed(1)+"%"],["Needed retry",(100*p.retry_rate).toFixed(1)+"%"],["GPU",g?g.util+"% · "+(g.mem_used/1024).toFixed(1)+"/"+(g.mem_total/1024).toFixed(0)+" GB · "+g.temp+"°C":"n/a"]];
$("tiles").innerHTML=T.map(t=>`<div class="tile"><b>${t[1]}</b><span>${t[0]}</span></div>`).join("");
const c=p.chunk_seconds||[];if(c.length>1){const mx=Math.max(...c)*1.1,pts=c.map((v,i)=>(i/(c.length-1)*300)+","+(60-v/mx*58)).join(" ");
$("spark").innerHTML=`<polyline fill="none" stroke="var(--acc)" stroke-width="1.5" points="${pts}"/>`;$("sparklbl").textContent="last "+c.length+" chunks · latest "+c[c.length-1]+"s · range "+Math.min(...c)+"–"+Math.max(...c)+"s"}
else $("sparklbl").textContent="waiting for the first chunk of this run…";
const L=Object.entries(p.labels||{}).sort((a,b)=>b[1]-a[1]),m=Math.max(1,...L.map(x=>x[1]));
$("labels").innerHTML=L.map(([k,v])=>`<div class="row"><span>${k}</span><i style="width:${v/m*60}%"></i><span>${v.toLocaleString()}</span></div>`).join("");
$("log").textContent=d.log_tail.join("\\n")}
tick();setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body, ctype = json.dumps(status()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Monitoring {ANN_DIR}\nOpen http://localhost:{args.port}  (Ctrl-C to stop; the annotation job is unaffected)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
