"""Deterministic held-out edit cases with reversible mixtures and known controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
import soundfile as sf

from source_labels import FOREGROUND


SR = 44100
ROOT = Path("runs/Benchmark/_data/audiocaps")
EVAL = Path("runs/audits/audiocaps_mix_pool4_eval_v1/results.json")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def read(item):
    path = Path(item["ref"])
    if sha(path) != item["sha"]:
        raise ValueError(f"Reference changed: {path}")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        divisor = math.gcd(SR, sr)
        x = signal.resample_poly(x, SR//divisor, sr//divisor)
    x = np.pad(x[:10*SR], (0, max(0, 10*SR-len(x))))
    if not np.isfinite(x).all() or abs(x).max() < 1e-4:
        raise ValueError(f"Bad source: {path}")
    return (x * (.32/abs(x).max())).astype("float32")


def save(path, x):
    sf.write(path, x, SR, subtype="FLOAT")
    return str(path)


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def embed_event(bg, stem, at):
    # Keep event RMS comparable to the background, with a fixed clipping bound.
    event = stem * min(.35/abs(stem).max(), 1.4125 * rms(bg[at:at+len(stem)])/max(rms(stem),1e-5))
    mix = bg.copy()
    mix[at:at+len(stem)] += event
    return mix, event


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    wavdir = out / "references"
    wavdir.mkdir()
    archived = json.loads(EVAL.read_text())["protocol"]
    train = jsonl(ROOT / "official_train_scale_v1/train_7070.jsonl")
    train_ids = {str(r["id"]) for r in train}
    train_y = {str(r["youtube_id"]) for r in train}
    train_hashes = {r[k] for r in train for k in ("sha256", "source_sha256") if k in r}
    test_csv = ROOT / "official_train_scale_v1/official_test.csv"
    with test_csv.open(encoding="utf-8-sig", newline="") as f:
        official_test = {row["audiocap_id"]: row for row in csv.DictReader(f)}
    order = lambda r: hashlib.sha256(("edit-37021:"+r["sha"]).encode()).hexdigest()
    pool = sorted([r for r in archived["audio"] if not r["exploratory"]], key=order)
    stems = [r for r in pool if r["row"] in FOREGROUND]
    backgrounds = [r for r in pool if r["row"] not in FOREGROUND]
    instruments = ["electric guitar","acoustic guitar","piano","harpsichord","violin",
                   "saxophone","flute","trumpet","harmonica","accordion","cello"]
    music = []
    for row in sorted(archived["music"], key=order):
        text = row["prompt"].lower()
        match = next((x for x in instruments if x in text),None)
        target = next((x for x in ("saxophone","violin","piano","flute") if x not in text),None)
        if match and target and not row["exploratory"] and Path(row["ref"]).stem.rsplit("_",1)[0] not in train_y:
            music.append((row,match,target))
    if len(stems) < 32 or len(backgrounds) < 48 or len(music) < 16:
        raise ValueError(f"Insufficient sources: {len(stems)=}, {len(backgrounds)=}, {len(music)=}")
    sources = backgrounds[:48]+stems[:32]+[x[0] for x in music[:16]]
    for item in sources:
        path = Path(item["ref"])
        if "audiocaps" in path.parts:
            raw = official_test[path.stem]
            item.update(sample_id=path.stem,youtube_id=raw["youtube_id"],start_time=raw["start_time"],split="test")
        else:
            ytid, start = path.stem.rsplit("_",1)
            item.update(sample_id=path.stem,youtube_id=ytid,start_time=start,split="MusicCaps held-out")
        if item["sha"] in train_hashes or item["sample_id"] in train_ids or item["youtube_id"] in train_y:
            raise ValueError(f"Post-training overlap: {item['sample_id']}")
    if len({x["sha"] for x in sources}) != len(sources):
        raise ValueError("Repeated held-out source")
    rows, data = [], []
    for i, (bg, front) in enumerate(zip(backgrounds[:32], stems[:32]),1):
        group = f"edit_{i:02d}"
        b, f = read(bg), read(front)
        hop = SR//2
        span_n = 3*SR
        crop = max(range(0, 7*SR+1, hop), key=lambda j:rms(f[j:j+span_n]))
        stem = f[crop:crop+span_n].copy()
        fades = SR//18
        stem[:fades] *= np.linspace(0,1,fades)
        stem[-fades:] *= np.linspace(1,0,fades)
        at = (2+i%4)*SR
        mix, event = embed_event(b, stem, at)
        bpath = save(wavdir/f"{group}_bg.wav",b)
        mpath = save(wavdir/f"{group}_mix.wav",mix)
        epath = save(wavdir/f"{group}_stem.wav",event)
        event_name, pann = FOREGROUND[front["row"]]
        common = dict(group=group, desc=event_name,event=event_name,panns_label=pann,
                      span=[at/SR,(at+len(stem))/SR], background=bpath, seed=37021+i,
                      parent_source=bg, parent_event=front, crop_start=crop/SR, stem=epath)
        for task, source, target in (("Add",bpath,mpath),("Remove",mpath,bpath)):
            add = task=="Add"
            rows.append(dict(common,id=f"{group}_{task.lower()}",task=task,kind=task.lower(),
                             source=source,target=target,
                             instruction=f"{task} the sound of '{event_name}' {'to' if add else 'from'} the input audio.",
                             target_description=f"{bg['prompt']} {event_name}." if add else bg["prompt"]))
        data.append((common,b,stem,at))
    for i,(orig,b,stem,at) in enumerate(data[:12]):
        candidates = data[16+i:] + data[16:16+i]
        other,different,alternate,_ = next(
            x for x in candidates if x[0]["event"] != orig["event"] and x[0]["panns_label"] != orig["panns_label"])
        group = orig["group"]
        for kind, x, cue, event in (("swap_instruction",b,other,alternate),("swap_source",different,orig,stem)):
            mixed,scaled_event = embed_event(x,event,at)
            path = save(wavdir/f"{group}_{kind}.wav",mixed)
            stem_path = save(wavdir/f"{group}_{kind}_stem.wav",scaled_event)
            background = orig["background"] if kind=="swap_instruction" else other["background"]
            rows.append(dict(orig,id=f"{group}_{kind}",kind=kind,task="Add",source=background,
                             target=path,desc=cue["desc"],event=cue["event"],background=background,
                             parent_id=f"{group}_add",panns_label=cue["panns_label"],
                             parent_source=(orig if kind=="swap_instruction" else other)["parent_source"],
                             parent_event=cue["parent_event"],crop_start=cue["crop_start"],stem=stem_path,
                             instruction=f"Add the sound of '{cue['desc']}' to the input audio."))
    for i,item in enumerate(backgrounds[32:48]):
        group = f"attr_{i:02d}"
        x = read(item)
        if i < 8:
            attribute = "lowband_eq"
            sos = signal.butter(4, 250, fs=SR, output="sos")
            y = x + (10**(-9/20)-1)*signal.sosfiltfilt(sos,x)
            instruction = "Reduce the bass in the input audio and keep the other sounds."
        else:
            attribute = "reverb"
            rng = np.random.default_rng(37021+i)
            t = np.arange(SR*3//4)/SR
            ir = rng.normal(size=len(t))*np.exp(-12*t)
            ir /= np.linalg.norm(ir)
            y = x + .30*signal.fftconvolve(x,ir)[:len(x)]
            instruction = "Add short room reverberation to the input audio and keep its original sounds."
        rows.append(dict(id=group,group=group,kind="attribute",task="Custom",attribute=attribute,
                         instruction=instruction,source=save(wavdir/f"{group}_src.wav",x),
                         target=save(wavdir/f"{group}_target.wav",y),parent_source=item,seed=37521+i))
    for i,(item,a,b) in enumerate(music[:16]):
        group = f"style_{i:02d}"
        rows.append(dict(id=group,group=group,kind="style",task="Style Transfer",
                         source_category=f"playing {a}",target_category=f"playing {b}",
                         instruction=f"Change the sound of 'playing {a}' to 'playing {b}'.",
                         source=save(wavdir/f"{group}_src.wav",read(item)),target=None,
                         parent_source=item,seed=37621+i))
    for row in rows:
        for field in ("source","target","background","stem"):
            if row.get(field):
                row[field+"_sha256"] = sha(row[field])
                if row[field+"_sha256"] in train_hashes:
                    raise ValueError("Derived audio collides with post-training source")
    manifest = out/"manifest.jsonl"
    manifest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    (out/"dataset_audit.json").write_text(json.dumps(dict(source_archive_sha256=sha(EVAL),
        train_manifest_sha256=sha(ROOT/"official_train_scale_v1/train_7070.jsonl"),
        cases=len(rows),sources=len(sources),manifest_sha256=sha(manifest),overlap={k:0 for k in ("sample_id","youtube_id","sha256")},
        foreground_selection="caption-reviewed candidates, maximum RMS 3 s crop; no human isolation ratings"),
        indent=2),encoding="utf-8")
    print(f"Prepared {len(rows)} cases, SHA256 {sha(manifest)}",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    prepare(parser.parse_args().out)
