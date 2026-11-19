"""Prepare a blinded quality pilot from the archived low-LR paired evaluation."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import re
import secrets
import shutil
import subprocess
import wave
import zipfile
from pathlib import Path


SOURCE = "852c5c459ec89f8b0a4e8ead80fa53fa8903e227b84c62e0d883362c24369b20"
ARMS = ("base", "pure_7070", "t2a_7070", "mix_7070")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def select(results):
    if digest(results) != SOURCE:
        raise ValueError("Unexpected source evaluation")
    data = json.loads(results.read_text(encoding="utf-8"))["generation"]
    selected = []
    for domain, setting in (("AudioCaps", "audio_10_cfg6"), ("MusicCaps", "music_10_cfg6")):
        rows = data[setting]["base"]["rows"]
        eligible = [idx for idx, row in enumerate(rows) if not row["exploratory"]]
        eligible.sort(key=lambda idx: hashlib.sha256(
            ("quality-pilot-v1:" + domain + ":" + rows[idx]["sha"]).encode()).digest())
        for idx in eligible[:8]:
            group = dict(domain=domain, sha=rows[idx]["sha"], prompt=rows[idx]["prompt"], clips=[])
            for arm in ARMS:
                row = data[setting][arm]["rows"][idx]
                if any(row[key] != rows[idx][key] for key in ("sha", "prompt", "seed", "frames", "source_sha", "exploratory")):
                    raise ValueError("Pairing mismatch")
                if not re.fullmatch(r"runs/audits/[a-zA-Z0-9_/]+/\d{3}\.wav", row["wav"]):
                    raise ValueError("Unexpected input path")
                group["clips"].append(dict(arm=arm, path=row["wav"], seed=row["seed"]))
            selected.append(group)
    if len(selected) != 16 or len({block["sha"] for block in selected}) != 16:
        raise ValueError("Expected 16 distinct paired sources")
    return selected


def measure(path):
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(path),
                           "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                          capture_output=True, text=True, check=True)
    match = re.search(r'\{\s*"input_i".*?\}', proc.stderr, re.S)
    meter = json.loads(match.group())
    loudness, peak = float(meter["input_i"]), float(meter["input_tp"])
    if not all(map(math.isfinite, (loudness, peak))):
        raise ValueError(f"Unmeasurable source: {path}")
    return loudness, peak


def inspect_clip(clip, source_dir):
    path = source_dir / clip["path"]
    with wave.open(str(path), "rb") as wav:
        if (wav.getnframes(), wav.getframerate(), wav.getsampwidth()) != (440320, 44100, 2):
            raise ValueError(f"Unexpected format: {path}")
        clip["channels"] = wav.getnchannels()
    clip["input_lufs"], clip["input_dbtp"] = measure(path)
    clip["source_sha256"] = digest(path)


def render(clip, source_dir, audio_dir, target):
    gain = target - clip["input_lufs"]
    filename = secrets.token_hex(10) + ".wav"
    out = audio_dir / filename
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source_dir / clip["path"]),
                    "-map_metadata", "-1", "-fflags", "+bitexact", "-af", f"volume={gain:.4f}dB",
                    "-c:a", "pcm_s16le", str(out)], check=True)
    loudness, peak = measure(out)
    with wave.open(str(out), "rb") as wav:
        if (wav.getnframes(), wav.getframerate(), wav.getnchannels()) != (440320, 44100, clip["channels"]):
            raise ValueError("Render changed duration, rate or channel layout")
    if abs(loudness - target) > 0.35 or peak > -0.9:
        raise ValueError(f"Loudness/peak out of tolerance: {out} {loudness} {peak}")
    clip.update(gain_db=gain, output_lufs=loudness, output_dbtp=peak,
                output_sha256=digest(out), src="audio/" + filename)


def build(selected, source_dir, root):
    if root.exists():
        raise FileExistsError(root)
    clips = [clip for block in selected for clip in block["clips"]]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda clip: inspect_clip(clip, source_dir), clips))
    target = min(-23.0, math.floor(10 * min(c["input_lufs"] - c["input_dbtp"] - 1.2 for c in clips)) / 10)
    master = root / "private"
    audio_dir = master / "audio"
    audio_dir.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda clip: render(clip, source_dir, audio_dir, target), clips))
    save(master / "sources.json", dict(results_sha256=SOURCE, target_lufs=target, blocks=selected))
    rng = random.SystemRandom()
    for domain in ("AudioCaps", "MusicCaps"):
        subset = [b for b in selected if b["domain"] == domain]
        rng.shuffle(subset)
        for offset in (0, 4):
            order = list(range(4))
            rng.shuffle(order)
            for shift in range(4):
                subset[offset+shift]["order"] = order[shift:] + order[:shift]
    rng.shuffle(selected)
    for form in ("A", "B"):
        folder = root / ("form-" + form)
        folder.mkdir()
        public = dict(version=1, pack_id="quality-v1-" + form + "-" + secrets.token_hex(4), blocks=[])
        key = dict(version=1, pack_id=public["pack_id"], blocks=[])
        for serial, block in enumerate(selected if form == "A" else reversed(selected), 1):
            block_id = f"Q{serial:02d}"
            visible, hidden = [], []
            order = block["order"] if form == "A" else list(reversed(block["order"]))
            for slot in order:
                clip = block["clips"][slot]
                clip_id = secrets.token_hex(6)
                visible.append(dict(id=clip_id, src=clip["src"]))
                hidden.append(dict(id=clip_id, arm=clip["arm"]))
            public["blocks"].append(dict(id=block_id, clips=visible))
            key["blocks"].append(dict(id=block_id, domain=block["domain"], sha=block["sha"], clips=hidden))
        save(master / ("key-" + form + ".json"), key)
        (folder / "pack.js").write_text("globalThis.LISTENING_PACK = " + json.dumps(public) + ";", encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("index.html"), folder / "index.html")
        shutil.copytree(audio_dir, folder / "audio")
        with zipfile.ZipFile(root / ("listening-" + form + ".zip"), "x", zipfile.ZIP_STORED) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(folder))
    print(json.dumps(dict(target_lufs=target, loudness=[min(c["output_lufs"] for c in clips), max(c["output_lufs"] for c in clips)],
                          max_dbtp=max(c["output_dbtp"] for c in clips), clips=len(clips))))


def add_prompt_forms(selected, root):
    master = root / "private"
    sources = json.loads((master / "sources.json").read_text(encoding="utf-8"))
    if sources["results_sha256"] != SOURCE:
        raise ValueError("Private source manifest is not from this evaluation")
    prompts = {(block["domain"], block["sha"]): block for block in selected}
    sources_by_sha = {(block["domain"], block["sha"]): block for block in sources["blocks"]}
    template = Path(__file__).with_name("index.html")
    prefix, flag = "globalThis.LISTENING_PACK = ", "\nglobalThis.HAS_ALIGNMENT = true;"
    for form in ("A", "B"):
        folder = root / ("form-" + form)
        key = json.loads((master / ("key-" + form + ".json")).read_text(encoding="utf-8"))
        manifest = folder / "pack.js"
        packed = manifest.read_text(encoding="utf-8").strip().removesuffix(flag)
        if not packed.startswith(prefix) or not packed.endswith(";"):
            raise ValueError("Unexpected public form format")
        public = json.loads(packed[len(prefix):-1])
        if public["pack_id"] != key["pack_id"] or len(public["blocks"]) != len(selected):
            raise ValueError("Form does not match key")
        captions, paths, seen = {}, [], set()
        for visible, hidden in zip(public["blocks"], key["blocks"], strict=True):
            ident = (hidden["domain"], hidden["sha"])
            reference = prompts[ident]
            if visible["id"] != hidden["id"] or ident in seen or visible["id"] in captions:
                raise ValueError("Duplicate or inconsistent block")
            seen.add(ident)
            sources_by_arm = {clip["arm"]: clip for clip in sources_by_sha[ident]["clips"]}
            original = {clip["arm"]: clip for clip in reference["clips"]}
            if {clip["arm"] for clip in hidden["clips"]} != set(ARMS) or len(visible["clips"]) != len(ARMS):
                raise ValueError("Unexpected arm assignments")
            for item, assignment in zip(visible["clips"], hidden["clips"], strict=True):
                source = sources_by_arm[assignment["arm"]]
                src = item["src"]
                if (item["id"] != assignment["id"] or source["src"] != src
                        or not re.fullmatch(r"audio/[a-f0-9]{20}\.wav", src)
                        or source["path"] != original[assignment["arm"]]["path"]
                        or source["seed"] != original[assignment["arm"]]["seed"]
                        or digest(folder / src) != source["output_sha256"]):
                    raise ValueError("Clip does not match its generated source")
                paths.append(src)
            captions[visible["id"]] = reference["prompt"]
        if len(set(paths)) != 64 or not all(isinstance(v, str) and v.strip() for v in captions.values()):
            raise ValueError("Incomplete listening form")
        (folder / "prompts.js").write_text("globalThis.LISTENING_PROMPTS = " +
            json.dumps({"pack_id": public["pack_id"], "prompts": captions}, ensure_ascii=True) + ";", encoding="utf-8")
        manifest.write_text(packed + flag, encoding="utf-8")
        for page in ("index.html", "alignment.html"):
            shutil.copyfile(template, folder / page)
        archive = root / ("listening-with-prompts-" + form + ".zip")
        temp = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_STORED) as bundle:
            for name in ["index.html", "alignment.html", "pack.js", "prompts.js", *paths]:
                bundle.write(folder / name, name)
        temp.replace(archive)
        print(f"{archive.name}: {len(captions)} original prompts, {len(paths)} verified clips")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("list", "build", "with-prompts"))
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    selection = select(args.results)
    if args.phase == "list":
        print("\n".join(c["path"] for b in selection for c in b["clips"]))
    elif args.phase == "with-prompts":
        if args.out is None:
            parser.error("with-prompts requires an existing --out form directory")
        add_prompt_forms(selection, args.out)
    else:
        if args.source_dir is None or args.out is None:
            parser.error("build requires --source-dir and --out")
        build(selection, args.source_dir, args.out)
