#!/usr/bin/env python3
"""Build an emoji gallery page from whatever is in its artifact folder.

    python3 emoji_gallery.py [sethmojis]

The folder (/home/wilson/artifacts/<set>/, served at http://wilson/<set>/) is the source of
truth: every file named <name>-<who><suffix> belongs to meme <name>, and the page is rebuilt from
the files, never from a hand-kept list. <set>.json next to this script carries what files cannot:
the blurb, per-meme notes and the retired list. sethmoji.py deliver() copies a meme's files in
and calls this, so a render shows up on the page the moment it is delivered.

Suffixes: "" emoji (.gif or .png) | -comparison | -full | -hires | -512 | -caption | anything else
is an extra. A meme with no emoji uses its first image as the grid tile.
"""
import glob
import html
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SET = sys.argv[1] if len(sys.argv) > 1 else "sethmojis"
ART = f"/home/wilson/artifacts/{SET}"
CFG = json.load(open(f"{HERE}/{SET}.json"))
WHO = CFG["who"]
IMG = (".gif", ".png", ".jpg", ".webp")


def collect():
    memes = {}
    for path in glob.glob(f"{ART}/*-{WHO}*"):
        f = os.path.basename(path)
        m = re.match(rf"^(.+?)-{WHO}(.*?)(\.[a-z0-9]+)$", f)
        if not m:
            continue
        name, suf, ext = m.groups()
        suf = re.sub(r"^-", "", suf)
        e = memes.setdefault(name, {"files": {}, "mtime": 0})
        e["files"][(suf, ext)] = f
        # order by the emoji itself: backfilling a full.gif must not move a meme to the top
        if suf in ("", "still-128") or not e["mtime"]:
            e["mtime"] = os.path.getmtime(path) if suf in ("", "still-128") else max(e["mtime"], os.path.getmtime(path))
    return memes


def role(files):
    pick = lambda *keys: next((files[k] for k in keys if k in files), None)
    r = {"emoji": pick(("", ".gif"), ("", ".png"), ("128", ".png"), ("still-128", ".png")),
         "comparison": pick(("comparison", ".gif")),
         "full": pick(("full", ".gif"), ("caption", ".gif"), ("full", ".png")),
         "hires": pick(("hires", ".mp4"), ("full", ".mp4"), ("caption", ".mp4"), ("", ".mp4"))}
    used = set(v for v in r.values() if v)
    r["extras"] = sorted(f for f in files.values() if f not in used)
    if not r["emoji"]:
        r["emoji"] = next((f for f in sorted(files.values()) if f.endswith(IMG)), None)
    return r


def main():
    memes = collect()
    retired = set(CFG.get("retired", []))
    notes = CFG.get("notes", {})
    items = []
    for name, e in sorted(memes.items(), key=lambda kv: -kv[1]["mtime"]):
        r = role(e["files"])
        if not r["emoji"]:
            continue
        v = int(max(os.path.getmtime(f"{ART}/{f}") for f in e["files"].values()))
        q = lambda f: f"{f}?v={v}" if f else None
        items.append({"name": name, "note": notes.get(name, ""), "retired": name in retired,
                      "emoji": q(r["emoji"]), "comparison": q(r["comparison"]),
                      "full": q(r["full"]), "hires": q(r["hires"]),
                      "extras": [q(x) for x in r["extras"]]})
    live = [i for i in items if not i["retired"]]
    page = TEMPLATE.replace("__TITLE__", html.escape(CFG["title"])) \
        .replace("__BLURB__", html.escape(CFG["blurb"])) \
        .replace("__COUNT__", str(len(live))) \
        .replace("__DATA__", json.dumps(items)) \
        .replace("__BUILT__", time.strftime("%Y-%m-%d %H:%M"))
    tmp = f"{ART}/index.html.tmp"
    open(tmp, "w").write(page)
    os.replace(tmp, f"{ART}/index.html")
    print(f"{SET}: {len(live)} live, {len(items) - len(live)} retired")


TEMPLATE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#141210;--fg:#e8e2d8;--mute:#9a9288;--line:#2a2724;--tile:#36393f;--acc:#ffcc00}
body.light{--bg:#f6f3ee;--fg:#2a2520;--mute:#7a7068;--line:#e2dcd2;--tile:#ffffff}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui;margin:0;padding:24px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;margin-bottom:16px}
h1{font-size:22px;margin:0}.count{color:var(--mute)}.blurb{color:var(--mute);margin:0 0 14px;max-width:900px}
.tools{margin-left:auto;display:flex;gap:8px}input,button{font:inherit;background:var(--tile);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px}
.cell{background:var(--tile);border:1px solid var(--line);border-radius:8px;padding:10px 6px 6px;text-align:center;cursor:pointer}
.cell:hover{outline:2px solid var(--acc)}.cell img{width:128px;height:128px;object-fit:contain;display:block;margin:0 auto}
.cell .n{font:12px ui-monospace,monospace;color:var(--mute);margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cell.retired{opacity:.45}h2{font-size:15px;color:var(--mute);margin:28px 0 10px}
dialog{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:0;max-width:min(1100px,96vw);width:100%}
dialog::backdrop{background:#000c}.dh{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--line)}
.dh b{font:600 16px ui-monospace,monospace;color:var(--acc)}.dh .x{margin-left:auto}.db{padding:16px;max-height:82vh;overflow:auto}
.db .note{color:var(--mute);margin:0 0 12px}.db h3{font-size:13px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em;margin:16px 0 6px}
.db img,.db video{max-width:100%;border-radius:6px;display:block}.sw{display:flex;gap:10px;align-items:center}.sw span{padding:10px;border-radius:6px;display:flex;align-items:center}
.dk{background:#36393f}.lt{background:#fff}.sw img{width:128px;height:auto}.dl a{color:var(--acc);margin-right:12px}
</style>
<header><h1>__TITLE__</h1><span class=count>__COUNT__ emojis</span>
<div class=tools><input id=q placeholder="filter…" autocomplete=off><button id=theme>light/dark</button></div></header>
<p class=blurb>__BLURB__</p>
<div class=grid id=grid></div><h2 id=rh hidden>Retired</h2><div class=grid id=rgrid></div>
<dialog id=dlg><div class=dh><b id=dt></b><span id=dpos class=count></span><button class=x id=prev>‹</button><button id=next>›</button><button id=close>✕</button></div><div class=db id=db></div></dialog>
<p class=count style="margin-top:28px">built __BUILT__ from the files in this folder</p>
<script>
const D=__DATA__;let cur=-1,shown=[];
const $=s=>document.querySelector(s);
function tile(i){const d=document.createElement('div');d.className='cell'+(D[i].retired?' retired':'');d.title=':'+D[i].name+':';
 d.innerHTML=`<img loading=lazy src="${D[i].emoji}"><div class=n>:${D[i].name}:</div>`;d.onclick=()=>open_(i);return d}
function render(){const f=$('#q').value.trim().toLowerCase();$('#grid').innerHTML='';$('#rgrid').innerHTML='';shown=[];
 D.forEach((m,i)=>{if(f&&!m.name.includes(f)&&!m.note.toLowerCase().includes(f))return;
  (m.retired?$('#rgrid'):$('#grid')).appendChild(tile(i));if(!m.retired)shown.push(i)});
 $('#rh').hidden=!$('#rgrid').children.length}
function media(src){return /\.mp4/.test(src)?`<video src="${src}" autoplay loop muted playsinline controls></video>`:`<img src="${src}">`}
function open_(i){cur=i;const m=D[i];$('#dt').textContent=':'+m.name+':';
 const p=shown.indexOf(i);$('#dpos').textContent=p>=0?`${p+1} / ${shown.length}`:'retired';
 let h=m.note?`<p class=note>${m.note}</p>`:'';
 h+=`<h3>Slack emoji</h3><div class=sw><span class=dk><img src="${m.emoji}"></span><span class=lt><img src="${m.emoji}"></span><span class=dk><img src="${m.emoji}" style="width:22px"></span></div>`;
 if(m.comparison)h+=`<h3>Side-by-side</h3>`+media(m.comparison);
 if(m.full)h+=`<h3>Full size</h3>`+media(m.full);
 if(m.hires)h+=`<h3>High resolution</h3>`+media(m.hires);
 m.extras.forEach(x=>h+=`<h3>${x.split('?')[0]}</h3>`+media(x));
 const all=[m.emoji,m.comparison,m.full,m.hires,...m.extras].filter(Boolean);
 h+=`<h3>Files</h3><p class=dl>`+all.map(x=>`<a href="${x}" download>${x.split('?')[0]}</a>`).join('')+`</p>`;
 $('#db').innerHTML=h;if(!$('#dlg').open)$('#dlg').showModal();history.replaceState(null,'','#'+m.name)}
function step(d){const p=shown.indexOf(cur);if(p<0)return;open_(shown[(p+d+shown.length)%shown.length])}
$('#prev').onclick=()=>step(-1);$('#next').onclick=()=>step(1);$('#close').onclick=()=>$('#dlg').close();
$('#dlg').addEventListener('close',()=>history.replaceState(null,'',location.pathname));
$('#dlg').addEventListener('click',e=>{if(e.target.id==='dlg')$('#dlg').close()});
document.addEventListener('keydown',e=>{if(!$('#dlg').open)return;if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)});
$('#q').oninput=render;$('#theme').onclick=()=>{document.body.classList.toggle('light');localStorage.theme=document.body.classList.contains('light')?'light':'dark'};
if(localStorage.theme==='light')document.body.classList.add('light');
render();const h=decodeURIComponent(location.hash.slice(1));const k=D.findIndex(m=>m.name===h);if(k>=0)open_(k);
</script>"""

if __name__ == "__main__":
    main()
