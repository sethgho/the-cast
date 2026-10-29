#!/usr/bin/env python3
"""Build an emoji gallery page from whatever is in its artifact folder.

    python3 emoji_gallery.py [sethmojis]

The folder (/home/wilson/artifacts/<set>/, served at http://wilson/<set>/) is the source of
truth: every file named <name>-<who><suffix> belongs to meme <name>, and the page is rebuilt from
the files, never from a hand-kept list. <set>.json next to this script carries what files cannot:
the blurb, per-meme notes and the retired list. sethmoji.py deliver() copies a meme's files in
and calls this, so a render shows up on the page the moment it is delivered.

Templates (<set>.json "templates", images under templates/ in the folder) drive the meme generator
section: the page draws the text onto the template in a canvas, client-side, and saves memes in the
browser's localStorage. Nothing under templates/ is scanned as a meme.

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
        .replace("__TPL__", json.dumps(CFG.get("templates", []))) \
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
.tools{margin-left:auto;display:flex;gap:8px}input,button,select{font:inherit;background:var(--tile);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
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
.dk{background:#36393f}.lt{background:#fff}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:0 0 16px}.tabs button{border:0;border-bottom:2px solid transparent;border-radius:0;background:none;color:var(--mute);padding:8px 14px;cursor:pointer;font-weight:600}
.tabs button.on{color:var(--fg);border-bottom-color:var(--acc)}.gw{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:20px;align-items:start}
@media(max-width:820px){.gw{grid-template-columns:1fr}}#cv{width:100%;max-width:640px;border-radius:6px;background:#fff;display:block}
.gf label{display:block;font-size:13px;color:var(--mute);margin:10px 0 4px}.gf textarea{width:100%;min-height:64px;font:inherit;background:var(--tile);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;resize:vertical}
.gf select{width:100%}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.saved{list-style:none;padding:0;margin:8px 0 0}
.saved li{display:flex;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid var(--line)}.saved li span{flex:1;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.saved li span:hover{color:var(--acc)}.ex{color:var(--mute);font-size:12px}.sw img{width:128px;height:auto}.dl a{color:var(--acc);margin-right:12px}
</style>
<header><h1>__TITLE__</h1><span class=count>__COUNT__ emojis</span>
<div class=tools><input id=q placeholder="filter…" autocomplete=off><button id=theme>light/dark</button></div></header>
<nav class=tabs><button data-tab=emoji class=on>__TITLE__</button><button data-tab=gen>Meme Generator</button></nav>
<section id=emoji><p class=blurb>__BLURB__</p>
<div class=grid id=grid></div><h2 id=rh hidden>Retired</h2><div class=grid id=rgrid></div></section>
<section id=gen hidden>
<p class=blurb>Pick a template, type the text for each panel. It draws in your browser; saved memes live in this browser only (localStorage). Download for the PNG.</p>
<div class=gw><div><canvas id=cv></canvas></div><div class=gf>
<label>Template</label><select id=tpl></select><div id=fields></div>
<div class=row><input id=title placeholder="name (optional)" style="flex:1"><button id=save>Save</button><button id=dl>Download PNG</button><button id=clear>New</button></div>
<h3 style="font-size:13px;color:var(--mute);margin:20px 0 0">Saved in this browser</h3><ul class=saved id=saved></ul>
<h3 style="font-size:13px;color:var(--mute);margin:16px 0 0">Examples</h3><ul class=saved id=examples></ul>
</div></div></section>
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

const T=__TPL__,KEY='sethmoji-memes',img={};let tpl=T[0],editing=null;
const load_=()=>{try{return JSON.parse(localStorage[KEY]||'[]')}catch(e){return[]}};
const store=v=>localStorage[KEY]=JSON.stringify(v);
function getImg(t){return img[t.id]||(img[t.id]=new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=t.image}))}
function wrap(ctx,text,maxw){const out=[];for(const para of text.split('\n')){let cur='';for(const w of para.split(/\s+/).filter(Boolean)){const t=cur?cur+' '+w:w;
 if(ctx.measureText(t).width<=maxw||!cur)cur=t;else{out.push(cur);cur=w}}out.push(cur)}return out}
async function draw(){if(!tpl)return;const i=await getImg(tpl),cv=$('#cv'),ctx=cv.getContext('2d');cv.width=i.naturalWidth;cv.height=i.naturalHeight;
 ctx.drawImage(i,0,0);ctx.fillStyle=tpl.color||'#000';ctx.textAlign='center';ctx.textBaseline='top';
 tpl.boxes.forEach((b,k)=>{const text=($('#f'+k)||{}).value||'';if(!text.trim())return;
  const X=b.x*cv.width,Y=b.y*cv.height,W=b.w*cv.width,H=b.h*cv.height;let size=Math.round(cv.height*0.06),lines,lh;
  for(;size>12;size-=2){ctx.font=`${size}px ${tpl.font}`;lines=wrap(ctx,text,W*0.84);lh=size*1.2;
   if(lh*lines.length<=H*0.8&&lines.every(l=>ctx.measureText(l).width<=W*0.9))break}
  if(tpl.stroke){ctx.strokeStyle=tpl.stroke;ctx.lineWidth=size*0.2;ctx.lineJoin='round'}
  let y=Y+(H-lh*lines.length)/2;lines.forEach(l=>{if(tpl.stroke)ctx.strokeText(l,X+W/2,y);ctx.fillText(l,X+W/2,y);y+=lh})})}
function fields(texts){$('#fields').innerHTML=tpl.boxes.map((b,k)=>`<label for=f${k}>${b.label}</label><textarea id=f${k}></textarea>`).join('');
 tpl.boxes.forEach((b,k)=>{const f=$('#f'+k);f.value=(texts||[])[k]||'';f.oninput=draw});draw()}
function pick(id,texts,title){tpl=T.find(t=>t.id===id)||T[0];$('#tpl').value=tpl.id;$('#title').value=title||'';fields(texts)}
function list(){const v=load_();$('#saved').innerHTML=v.length?'':'<li class=ex>nothing saved yet</li>';
 v.forEach((m,k)=>{const li=document.createElement('li');li.innerHTML=`<span title="${m.texts.join(' / ')}">${m.title||m.texts.join(' / ')}</span><button>✕</button>`;
  li.querySelector('span').onclick=()=>{editing=k;pick(m.template,m.texts,m.title)};
  li.querySelector('button').onclick=()=>{const w=load_();w.splice(k,1);store(w);editing=null;list()};$('#saved').appendChild(li)});
 $('#examples').innerHTML='';T.forEach(t=>(t.examples||[]).forEach(e=>{const li=document.createElement('li');
  li.innerHTML=`<span>${t.name}: ${e.title}</span>`;li.querySelector('span').onclick=()=>{editing=null;pick(t.id,e.texts,'')};$('#examples').appendChild(li)}))}
if(T.length){$('#tpl').innerHTML=T.map(t=>`<option value="${t.id}">${t.name}</option>`).join('');
 $('#tpl').onchange=()=>{editing=null;pick($('#tpl').value,[], '')};
 $('#save').onclick=()=>{const m={template:tpl.id,title:$('#title').value.trim(),texts:tpl.boxes.map((b,k)=>$('#f'+k).value),saved:Date.now()};
  if(!m.texts.some(x=>x.trim()))return;const v=load_();if(editing!==null&&v[editing])v[editing]=m;else{v.unshift(m);editing=0}store(v);list()};
 $('#dl').onclick=async()=>{await draw();const a=document.createElement('a');const nm=($('#title').value.trim()||tpl.boxes.map((b,k)=>$('#f'+k).value).join(' ')).toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'').slice(0,60)||'meme';
  a.download=`${tpl.id}-${nm}.png`;a.href=$('#cv').toDataURL('image/png');a.click()};
 $('#clear').onclick=()=>{editing=null;pick(tpl.id,[],'')};
 pick(T[0].id,[],'');list()}else document.querySelector('[data-tab=gen]').hidden=true;
function tab(t,push){document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.tab===t));
 $('#emoji').hidden=t!=='emoji';$('#gen').hidden=t!=='gen';$('#q').style.visibility=t==='emoji'?'visible':'hidden';
 if(t==='gen')draw();if(push)history.replaceState(null,'',t==='gen'?'#generator':location.pathname)}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>tab(b.dataset.tab,true));
render();const h=decodeURIComponent(location.hash.slice(1));
if(h==='generator')tab('gen');else{const k=D.findIndex(m=>m.name===h);if(k>=0)open_(k)}
</script>"""

if __name__ == "__main__":
    main()
