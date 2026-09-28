#!/usr/bin/env python3
"""Fetch candidate source clips for sethmojis from Tenor: three per meme, for picking by eye.

    python3 sethmoji_source.py            # all
    python3 sethmoji_source.py blink nod  # some

Writes <OUT>/<name>/c<k>.mp4 plus a contact sheet per meme. Tenor's view page carries the clip
as a schema.org contentUrl; the mp4 variant is the one ending in "AAPo".
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

OUT = "/home/wilson/scratch/sethmoji"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
MEMES = {
    "blink": "blinking guy",
    "nod": "robert redford nod",
    "jim-look": "jim halpert looking at camera",
    "side-eye": "side eye chloe",
    "thats-me": "leonardo dicaprio pointing",
    "nooo": "michael scott no god please no",
    "lost": "confused travolta",
    "cheers": "leonardo dicaprio cheers gatsby",
    # batch 2
    "mind-blown": "tim heidecker mind blown",
    "slow-clap": "citizen kane slow clap",
    "mic-drop": "obama mic drop",
    "its-happening": "ron paul its happening",
    "popcorn": "michael jackson popcorn",
    "salty": "salt bae",
    "thumbs-up": "terminator thumbs up lava",
    "elaine": "elaine dance seinfeld",
}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def views(query, n=3):
    html = get("https://tenor.com/search/" + query.replace(" ", "-") + "-gifs")
    seen = []
    for m in re.finditer(r'href="(/view/[a-z0-9-]+-\d+)"', html):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen[:n]


def mp4_of(view):
    html = get("https://tenor.com" + view).replace("\\u002F", "/")
    for m in re.finditer(r'"contentUrl":"(https://media[0-9]*\.tenor\.com/[^"]+\.mp4)"', html):
        return m.group(1)
    return None


def main(names):
    for name in names:
        d = f"{OUT}/{name}"
        os.makedirs(d, exist_ok=True)
        for k, v in enumerate(views(MEMES[name])):
            url = mp4_of(v)
            if not url:
                continue
            dest = f"{d}/c{k}.mp4"
            subprocess.run(["curl", "-sL", "-m", "30", "-A", UA["User-Agent"], "-o", dest, url], check=False)
            json.dump({"view": v, "url": url}, open(f"{d}/c{k}.json", "w"))
        # sheet: 5 frames from each candidate, one row per candidate
        rows = []
        for k in range(3):
            src = f"{d}/c{k}.mp4"
            if not os.path.exists(src) or os.path.getsize(src) < 1000:
                continue
            row = f"{d}/row{k}.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf",
                            "thumbnail=4,scale=-2:120,tile=6x1", "-frames:v", "1", row], check=False)
            if os.path.exists(row):
                rows.append(row)
        if rows:
            args = ["ffmpeg", "-y", "-loglevel", "error"]
            for r in rows:
                args += ["-i", r]
            args += ["-filter_complex", f"vstack=inputs={len(rows)}" if len(rows) > 1 else "null",
                     f"{d}/sheet.png"]
            subprocess.run(args, check=False)
        print(name, [os.path.getsize(f"{d}/c{k}.mp4") // 1024 for k in range(3) if os.path.exists(f"{d}/c{k}.mp4")], "KB")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MEMES))
