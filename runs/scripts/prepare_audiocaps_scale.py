"""Prepare nested official-train AudioCaps pools and paired T2A schedules."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import time

import pyarrow.parquet as pq
import requests
import soundfile as sf

from reca_audio.train.gradient_experiment import load_schedule, write_schedule

DATA = Path("runs/Benchmark/_data")
OLD = Path("runs/train/reca_long_matrix_b_10s_v1/t2a_schedule.jsonl")
OFFICIAL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset"
REPO = "OpenSound/AudioCaps"
REVISION = "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"
ENDPOINT = "https://hf-mirror.com"
SEED = 42


def sha(data):
    return hashlib.sha256(data).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(session, url, path):
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=(15, 75)) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for chunk in response.iter_content(2**20):
                        handle.write(chunk)
            return
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def prepare(out, count, small_count):
    out.mkdir(parents=True, exist_ok=False)
    audio_dir = out / "audio"
    audio_dir.mkdir()
    blocked_videos, blocked_ids, blocked_hashes = set(), set(), set()
    official = {}
    sources = {}
    with requests.Session() as session:
        for split in ("train", "val", "test"):
            response = session.get(f"{OFFICIAL}/{split}.csv", timeout=45)
            response.raise_for_status()
            (out / f"official_{split}.csv").write_bytes(response.content)
            sources[split] = sha(response.content)
            records = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
            if split == "train":
                official = {row["audiocap_id"]: row for row in records}
            else:
                blocked_videos.update(row["youtube_id"] for row in records)
                blocked_ids.update(row["audiocap_id"] for row in records)
        existing_test = DATA / "audiocaps/test.csv"
        with existing_test.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                blocked_videos.add(row["youtube_id"])
                blocked_ids.add(row["audiocap_id"])
        music_root = DATA / "musiccaps"
        blocked_videos.update(row["id"].rsplit("_", 1)[0] for row in jsonl(music_root / "prompts1000.jsonl"))
        references = {p for p in (music_root / "ref").iterdir() if p.is_file()}
        for split in ("val", "test"):
            manifest = DATA / "audiocaps/reca_split_v1" / f"{split}_manifest.jsonl"
            references.update(Path(row["audio"]) for row in jsonl(manifest))
        for path in references:
            with path.open("rb") as handle:
                blocked_hashes.add(hashlib.file_digest(handle, "sha256").hexdigest())

        response = session.get(f"{ENDPOINT}/api/datasets/{REPO}/revision/{REVISION}", timeout=45)
        response.raise_for_status()
        metadata = response.json()
        if metadata["sha"] != REVISION:
            raise ValueError("dataset revision mismatch")
        shards = sorted(item["rfilename"] for item in metadata["siblings"]
                        if item["rfilename"].startswith("data/train-") and item["rfilename"].endswith(".parquet"))
        random.Random(SEED).shuffle(shards)
        if not shards:
            raise ValueError("no official train shards")
        seen_ids, seen_hashes, seen_videos = set(), set(), set()
        selected, used_shards = [], []
        rejected = Counter()
        with tempfile.TemporaryDirectory(prefix="reca-audiocaps-", dir=out) as scratch:
            parquet = Path(scratch) / "shard.parquet"
            for shard in shards:
                download(session, f"{ENDPOINT}/datasets/{REPO}/resolve/{REVISION}/{shard}", parquet)
                used_shards.append(shard)
                for batch in pq.ParquetFile(parquet).iter_batches(
                    batch_size=16, columns=["audiocap_id", "youtube_id", "start_time", "audio"],
                ):
                    for row in batch.to_pylist():
                        sample_id, video = str(row["audiocap_id"]), row["youtube_id"]
                        source = official.get(sample_id)
                        if (source is None or source["youtube_id"] != video
                                or int(source["start_time"]) != int(row["start_time"])):
                            rejected["not_verified_official_train"] += 1
                            continue
                        if sample_id in blocked_ids or video in blocked_videos:
                            rejected["heldout_id_or_video"] += 1
                            continue
                        if sample_id in seen_ids or video in seen_videos:
                            rejected["duplicate_id_or_video"] += 1
                            continue
                        raw = row["audio"]["bytes"]
                        if not raw:
                            rejected["missing_audio"] += 1
                            continue
                        digest = sha(raw)
                        if digest in blocked_hashes or digest in seen_hashes:
                            rejected["duplicate_or_heldout_sha"] += 1
                            continue
                        try:
                            with sf.SoundFile(io.BytesIO(raw)) as sound:
                                if not 1 <= sound.frames / sound.samplerate <= 10.5:
                                    rejected["duration"] += 1
                                    continue
                                audio = sound.read(dtype="float32")
                                rate = sound.samplerate
                        except (RuntimeError, ValueError):
                            rejected["undecodable"] += 1
                            continue
                        path = audio_dir / f"{sample_id}.flac"
                        sf.write(path, audio, rate, subtype="PCM_16")
                        saved_hash = sha(path.read_bytes())
                        if saved_hash in blocked_hashes or saved_hash in seen_hashes:
                            path.unlink()
                            rejected["duplicate_or_heldout_export_sha"] += 1
                            continue
                        selected.append(dict(audio=str(path), caption=source["caption"], id=sample_id,
                                             youtube_id=video, start_time=int(source["start_time"]),
                                             sha256=saved_hash, source_sha256=digest))
                        seen_ids.add(sample_id)
                        seen_videos.add(video)
                        seen_hashes.update((digest, saved_hash))
                        if len(selected) == count:
                            break
                    if len(selected) == count:
                        break
                print(f"{shard}: accepted {len(selected)}/{count}", flush=True)
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"only acquired {len(selected)}/{count} audio groups")
    assert not (seen_ids & blocked_ids or seen_videos & blocked_videos or seen_hashes & blocked_hashes)
    random.Random(SEED).shuffle(selected)
    original = load_schedule(OLD, expected_digest=OLD.with_suffix(".jsonl.sha256").read_text().strip())
    if len(original) != 2000 or any(row["task"] != "t2a" for row in original):
        raise ValueError("historical schedule is not the expected T2A control")
    pools = {}
    for n in (small_count, count):
        manifest = out / f"train_{n}.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected[:n]),
                            encoding="utf-8")
        schedule = out / f"schedule_{n}.jsonl"
        rng = random.Random(SEED)
        records = [dict(row, sample_index=int(rng.random() * n)) for row in original]
        digest = write_schedule(schedule, records)
        schedule.with_suffix(".jsonl.sha256").write_text(digest + "\n", encoding="utf-8")
        pools[n] = dict(manifest=str(manifest), manifest_sha256=sha(manifest.read_bytes()),
                        schedule=str(schedule), schedule_sha256=digest,
                        unique_exposed=len({row["sample_index"] for row in records}))
    save_json(out / "audit.json", dict(dataset=REPO, revision=REVISION, seed=SEED,
                                     official_csv_sha256=sources, shards=used_shards, pools=pools,
                                     heldout_intersection={"id": 0, "youtube_id": 0, "sha256": 0},
                                     blocked_video_count=len(blocked_videos),
                                     blocked_reference_hashes=len(blocked_hashes), rejected=dict(rejected)))
    print(json.dumps(pools), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=7070)
    parser.add_argument("--small-count", type=int, default=707)
    args = parser.parse_args()
    if not 0 < args.small_count < args.count:
        parser.error("require 0 < small-count < count")
    prepare(args.out, args.count, args.small_count)
