"""Score the fixed edit benchmark, keeping edit success and preservation separate."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import get_clap, get_panns


ARMS = ("base", "pure_7070", "t2a_7070", "mix_7070", "mix_pool4_7070", "text_slots_7070")
CONTRASTS = [(arm, "base") for arm in ARMS[1:]] + [
    ("mix_7070", "t2a_7070"), ("mix_7070", "text_slots_7070"), ("mix_pool4_7070", "mix_7070")
]
SR = 44100
N = 10 * SR
EVENT_CLAP_MIN = .025
EVENT_PANNS_MIN = .02
PRESERVATION_MIN = .10
PROGRESS_PASS = .5


def read_audio(path, *, generated=False):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SR or len(audio) < N or (generated and len(audio) != 483328):
        raise ValueError(f"Invalid shape or sample rate at {path}: {audio.shape}, {rate}")
    if not np.isfinite(audio).all() or not audio.any():
        raise ValueError(f"Invalid or silent signal: {path}")
    tail = audio[N:]
    quality = dict(frames=len(audio), over_full_scale=float(np.mean(np.abs(audio) > 1)),
                   tail_rms=float(np.sqrt(np.mean(tail**2))) if len(tail) else 0)
    return audio[:N].mean(axis=1), quality


def normalized(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def waveform_part(audio, span, role):
    if role == "full":
        return audio
    start, end = np.rint(np.asarray(span) * SR).astype(int)
    if not 0 <= start < end <= N:
        raise ValueError(f"Invalid edit span: {span}")
    if role == "event":
        return audio[start:end]
    if role != "preserve":
        raise ValueError(role)
    guard = SR // 4
    return np.concatenate((audio[:max(0, start-guard)], audio[min(N, end+guard):]))


class Features:
    def __init__(self, device, directory):
        self.device = device
        self.directory = directory
        self.clap = get_clap(device)
        # AudioTagging checks the literal "cuda" and uses visible CUDA devices.
        self.panns = get_panns("cuda" if device == "cuda:0" else device)
        self.labels = {str(label).casefold(): i for i, label in enumerate(self.panns.labels)}
        self.paths = {}
        self.cache = {}
        self.texts = {}

    def prepare(self, path, span, role):
        key = str(path), tuple(span), role
        if key not in self.cache and key not in self.paths:
            waveform, _ = read_audio(path)
            part = waveform_part(waveform, span, role)
            if len(part) < SR // 3:
                raise ValueError(f"Region too short to score at {path}: {span} {role}")
            temp = self.directory / f"{len(self.paths):07d}.wav"
            sf.write(temp, part, SR, subtype="FLOAT")
            self.paths[key] = temp
        return key

    def run(self, keys, panns_keys):
        todo = sorted(set(keys) - self.cache.keys())
        for i in range(0, len(todo), 16):
            part = todo[i:i+16]
            wavs = [str(self.paths[key]) for key in part]
            vecs = self.clap.get_audio_embedding_from_filelist(x=wavs, use_tensor=False)
            self.cache.update({k: {"clap": np.asarray(v, dtype=np.float32)} for k,v in zip(part,vecs)})
        # Preserve true clip duration. Group equal lengths for PANNs batching.
        work = defaultdict(list)
        for key in sorted(set(panns_keys)):
            if "panns" in self.cache[key]:
                continue
            wav, _ = sf.read(self.paths[key], dtype="float32")
            wav = signal.resample_poly(wav, 320, 441)
            work[len(wav)].append((key, wav))
        for items in work.values():
            for i in range(0, len(items), 8):
                part = items[i:i+8]
                probs, embeds = self.panns.inference(np.stack([wav for _,wav in part]))
                for (key,_), prob, embed in zip(part,probs,embeds):
                    self.cache[key].update(panns=prob, panns_embed=embed)

    def prime_text(self, texts):
        texts = sorted(set(texts) - self.texts.keys())
        for i in range(0, len(texts), 32):
            part = texts[i:i+32]
            vectors = self.clap.get_text_embedding(part, use_tensor=False)
            self.texts.update(zip(part, normalized(vectors)))

    def event(self, key, description, panns_label=None):
        vec = self.cache[key]
        clap = float(normalized(vec["clap"]) @ self.texts[description])
        label = self.labels.get(str(panns_label).casefold())
        panns = None if label is None or "panns" not in vec else float(vec["panns"][label])
        return clap, panns

    def similarity(self, a, b):
        return float(normalized(self.cache[a]["clap"]) @ normalized(self.cache[b]["clap"]))


def sisdr(estimate, reference):
    a, b = estimate.astype("float64"), reference.astype("float64")
    a -= a.mean()
    b -= b.mean()
    if float(b @ b) < 1e-9:
        return None
    proj = float(a @ b) / float(b @ b) * b
    err = a - proj
    return float(10 * np.log10(max(float(proj @ proj),1e-12)/max(float(err @ err),1e-12)))


def alignment_offset(estimate, reference, span):
    dec = 8
    est, ref = (signal.resample_poly(x, 1, dec) for x in (estimate, reference))
    low, high = np.rint(np.asarray(span) * SR / dec).astype(int)
    ref[max(0,low-1400):high+1400] = 0
    cross = signal.correlate(est - est.mean(), ref - ref.mean(), mode="full", method="fft")
    bound = int(.05 * SR / dec)
    center = len(ref)-1
    return int((np.argmax(cross[center-bound:center+bound+1]) - bound)*dec)


def log_mel(x):
    _, _, spec = signal.stft(x, fs=SR, nperseg=1024, noverlap=768)
    return np.log10(np.maximum(MEL @ np.abs(spec)**2, 1e-8))


def mel_matrix():
    hz = np.fft.rfftfreq(1024, 1/SR)
    mel = 2595*np.log10(1+hz/700)
    edges = np.linspace(0, mel[-1], 66)
    lo = (mel[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None])
    hi = (edges[2:,None]-mel[None,:])/(edges[2:,None]-edges[1:-1,None])
    return np.maximum(0, np.minimum(lo, hi)).astype("float32")


MEL = mel_matrix()


def log_mel_distance(a, b):
    return float(np.mean(np.abs(log_mel(a)-log_mel(b))))


def relative_progress(value, start, finish, margin):
    if value is None or start is None or finish is None or abs(finish-start) < margin:
        return None
    return (value-start)/(finish-start)


def passed(values):
    kept = [v for v in values if v is not None]
    return float(np.mean(kept) >= PROGRESS_PASS) if kept else None


def background_score(feat, output, background, distractor):
    if distractor is None:
        return feat.similarity(output, background), None, None
    own = feat.similarity(output, background)
    alt = feat.similarity(output, distractor)
    unmatched = feat.similarity(background,distractor)
    return own, relative_progress(own, unmatched, 1, PRESERVATION_MIN), own-alt


def typed_stats(estimate, reference, span):
    keep_est,keep_ref = (waveform_part(x, span,"preserve") for x in (estimate,reference))
    lag = alignment_offset(estimate, reference, span)
    # Only round-trip alignment within 50 ms is allowed. Zero fill, no wraparound.
    aligned = np.zeros_like(estimate)
    if lag >= 0:
        aligned[:len(estimate)-lag] = estimate[lag:]
    else:
        aligned[-lag:] = estimate[:len(estimate)+lag]
    return dict(preserve_sisdr=sisdr(keep_est,keep_ref),
                preserve_sisdr_aligned=sisdr(waveform_part(aligned, span,"preserve"),keep_ref),
                preserve_logmel=log_mel_distance(keep_est,keep_ref),
                estimated_lag_ms=1000*lag/SR)


def frechet(a, b):
    # Exact low-rank form for singular empirical covariances, common with <100 samples.
    a, b = (np.asarray(x, dtype="float64") for x in (a, b))
    mean = a.mean(0)-b.mean(0)
    ac = (a-a.mean(0))/np.sqrt(len(a)-1)
    bc = (b-b.mean(0))/np.sqrt(len(b)-1)
    cross = np.linalg.svd(ac@bc.T, compute_uv=False)
    return float(max(0., mean@mean + np.sum(ac**2)+np.sum(bc**2)-2*cross.sum()))


def distribution(generated, target):
    p, q = [np.maximum(np.stack([r["panns"] for r in v]).astype("float64"), 1e-10)
            for v in (generated, target)]
    p /= p.sum(1, keepdims=True)
    q /= q.sum(1, keepdims=True)
    return dict(clap_fad=frechet([r["clap"] for r in generated],[r["clap"] for r in target]),
                panns_fd=frechet([r["panns_embed"] for r in generated],[r["panns_embed"] for r in target]),
                panns_kl=float(np.mean(np.sum(p * np.log(p/q.mean(0)),axis=1))),
                panns_is=float(np.exp(np.mean(np.sum(p * np.log(p/p.mean(0)),axis=1)))))


def paired(lhs, rhs, key, *, by_task=False):
    groups = defaultdict(list)
    task_groups = defaultdict(dict)
    for a, b in zip(lhs, rhs, strict=True):
        if a["id"] != b["id"] or a["group"] != b["group"]:
            raise ValueError("Unpaired sample identity")
        if a.get(key) is not None and b.get(key) is not None:
            delta = a[key]-b[key]
            groups[a["group"]].append(delta)
            if by_task:
                task_groups[a["kind"]][a["group"]]=delta
    if not groups:
        return None
    draws=np.random.default_rng(42).integers(len(groups),size=(10000,len(groups)))
    if by_task:
        vals=np.array([[items.get(g,np.nan) for g in groups] for items in task_groups.values()])
        counts=np.isfinite(vals[:,draws]).sum(axis=-1)
        sampled=np.nansum(vals[:,draws],axis=-1)
        valid=(counts>0).all(0)
        boot=(sampled[:,valid]/counts[:,valid]).mean(0)
        estimate=float(np.nanmean(vals,axis=1).mean())
    else:
        vals=np.array([np.mean(v) for v in groups.values()])
        boot=vals[draws].mean(1)
        estimate=float(vals.mean())
    return dict(n=sum(map(len,groups.values())), groups=len(groups), delta=estimate,
                ci95=np.quantile(boot,[.025,.975]).tolist())


def resolve(root, value):
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return path.resolve()
    path = root/path
    return path.resolve(strict=True)


def case_span(row):
    return row.get("span") or [0,10]


def description(row):
    return row.get("event") or row.get("desc") or row.get("target_category")


def label(row):
    return row.get("panns_label") or row.get("label")


def lower_band_ratio(audio):
    hz, psd = signal.welch(audio,fs=SR,nperseg=4096)
    low = np.sum(psd[(hz>=40)&(hz<250)])
    mid = np.sum(psd[(hz>=250)&(hz<4000)])
    return float(10*np.log10(max(low,1e-12)/max(mid,1e-12)))


def native_row(feat, row, wav, roles, target, source, background, distractor):
    span = case_span(row)
    kind = row["kind"]
    event_region = "event" if kind == "remove" else "full"
    event_key = roles[event_region]
    txt = description(row)
    src_scores = feat.event(source[event_region], txt, label(row))
    ref_scores = feat.event(target[event_region], txt, label(row))
    got_scores = feat.event(event_key, txt, label(row))
    sign = -1 if kind == "remove" else 1
    limits = (EVENT_CLAP_MIN, EVENT_PANNS_MIN)
    progress = [
        relative_progress(out, start, end, margin)
        if start is not None and end is not None and sign*(end-start)>=margin else None
        for out,start,end,margin in zip(got_scores,src_scores,ref_scores,limits)
    ]
    edit = passed(progress)
    similarity, preservation_progress, background_margin = background_score(
        feat, roles["preserve"], background, distractor
    )
    preservation = (float(preservation_progress >= PROGRESS_PASS and background_margin > 0)
                    if preservation_progress is not None else None)
    ref_wav, _ = read_audio(row["background"])
    tgt_wav, _ = read_audio(row["target"])
    return dict(edit_success=edit, event_clap=got_scores[0], event_panns=got_scores[1],
                event_clap_reference_delta=ref_scores[0]-src_scores[0],
                event_panns_reference_delta=(ref_scores[1]-src_scores[1]
                                             if ref_scores[1] is not None and src_scores[1] is not None else None),
                event_progress_clap=progress[0], event_progress_panns=progress[1],
                edit_region_logmel=log_mel_distance(waveform_part(wav,span,"event"),
                                                     waveform_part(tgt_wav,span,"event")),
                preserve_embeddings=similarity, preserve_progress=preservation_progress,
                preserve_success=preservation, preserve_reference_margin=background_margin,
                joint_success=edit*preservation if edit is not None and preservation is not None else None,
                **typed_stats(wav, ref_wav, span))


def instruction_control(feat, old, changed, old_ref, new_ref, first_text, second_text):
    def preference(clip):
        return feat.event(clip, first_text)[0]-feat.event(clip, second_text)[0]
    first, second = preference(old), preference(changed)
    oracle_shift = preference(old_ref)-preference(new_ref)
    actual_shift = first-second
    valid = oracle_shift >= EVENT_CLAP_MIN
    return dict(instruction_switch_margin=actual_shift/2, instruction_reference_shift=oracle_shift/2,
                instruction_switch_progress=actual_shift/oracle_shift if valid else None,
                instruction_switch_success=(float(actual_shift>=PROGRESS_PASS*oracle_shift)
                                            if valid else None),
                instruction_switch_both_preferred=float(first>0 and second<0))


def source_control(feat, old, changed, first_bg, second_bg):
    first = feat.similarity(old,first_bg)-feat.similarity(old,second_bg)
    second = feat.similarity(changed,second_bg)-feat.similarity(changed,first_bg)
    separation = 1-feat.similarity(first_bg,second_bg)
    valid = separation >= PRESERVATION_MIN
    return dict(source_switch_margin=(first+second)/2, source_reference_margin=separation,
                source_switch_success=(float(first>=PROGRESS_PASS*separation and second>=PROGRESS_PASS*separation)
                                       if valid else None))


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),encoding="utf-8")
    tmp.replace(path)


def average_rows(rows):
    keys = sorted({k for r in rows for k,v in r.items() if isinstance(v,(float,int))})
    result = {}
    for k in keys:
        values = [r[k] for r in rows if r.get(k) is not None]
        result[k] = dict(n=len(values),mean=float(np.mean(values)))
    return result


def model_summary(rows):
    groups = {kind:[r for r in rows if r["kind"]==kind]
              for kind in ("add","remove","attribute","style","swap_instruction","swap_source")}
    groups["native"] = groups["add"]+groups["remove"]
    available_joint = [r for r in groups["native"] if r.get("joint_success") is not None]
    contingency = dict(n=len(available_joint),both=0,edit_only=0,preserve_only=0,neither=0)
    for r in available_joint:
        if r["joint_success"]:
            key = "both"
        elif r["edit_success"]:
            key = "edit_only"
        elif r["preserve_success"]:
            key = "preserve_only"
        else:
            key = "neither"
        contingency[key] += 1
    summaries={k:average_rows(v) for k,v in groups.items() if v}
    for key,d in summaries.get("native",{}).items():
        available=[summaries.get(k,{}).get(key) for k in ("add","remove")]
        means=[v["mean"] for v in available if v]
        d["mean"]=float(np.mean(means))
    return dict(groups=summaries,
                native_contingency_table=contingency)


def bootstrap(models):
    kinds = {
        "native": ("edit_success","preserve_success","joint_success","event_progress_clap",
                   "event_progress_panns","preserve_progress","preserve_logmel","preserve_sisdr",
                   "preserve_sisdr_aligned","preserve_embeddings","edit_region_logmel"),
        "add": ("edit_success","preserve_success","joint_success","event_clap","event_panns"),
        "remove": ("edit_success","preserve_success","joint_success","event_clap","event_panns"),
        "attribute": ("attribute_logmel_preference","attribute_band_progress","attribute_success"),
        "style": ("style_text_margin","style_target_clap","onset_envelope_correlation"),
        "swap_instruction": ("instruction_switch_margin","instruction_switch_progress",
                             "instruction_switch_success","instruction_switch_both_preferred"),
        "swap_source": ("source_switch_margin","source_switch_success"),
    }
    result = {}
    for a,b in CONTRASTS:
        contrast = {}
        for kind, fields in kinds.items():
            accept = {"add","remove"} if kind=="native" else {kind}
            lhs,rhs = ([r for r in models[name] if r["kind"] in accept] for name in (a,b))
            contrast[kind] = {key: paired(lhs,rhs,key,by_task=kind=="native") for key in fields}
        result[f"{a}__minus__{b}"] = contrast
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    root=args.out.resolve()
    rows=[json.loads(line) for line in (root/"manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids={r["id"] for r in rows}
    if not rows or len(ids)!=len(rows):
        raise ValueError("Missing or duplicate manifest IDs")
    complete=json.loads((root/"GENERATION_COMPLETE").read_text(encoding="utf-8"))
    if complete["rows"] != len(rows) or set(complete["arms"]) != set(ARMS):
        raise ValueError("Generation completion record disagrees with the manifest")
    for row in rows:
        for key in ("source","target","background","distractor_background"):
            row[key]=resolve(root,row.get(key))
        row["group"] = row.get("group",row["id"])
    outputs={}
    quality={}
    for arm in ARMS:
        folder=root/"audio"/arm
        if {p.stem for p in folder.glob("*.wav")}!=ids:
            raise ValueError(f"Incomplete benchmark for {arm} at {folder}")
        outputs[arm]={r["id"]:folder/(r["id"]+".wav") for r in rows}
        quality[arm]={i:read_audio(path,generated=True)[1] for i,path in outputs[arm].items()}
        for row in rows:
            path=outputs[arm][row["id"]]
            meta=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            with path.open("rb") as fh:
                digest=hashlib.file_digest(fh,"sha256").hexdigest()
            if ((meta["id"],meta["seed"],meta["frames"],meta["wav_sha256"]) !=
                    (row["id"],row["seed"],quality[arm][row["id"]]["frames"],digest)):
                raise ValueError(f"Generated WAV metadata or checksum changed at {path}")

    with tempfile.TemporaryDirectory(prefix="reca_edit_score_") as temp:
        feat = Features(args.device,Path(temp))
        if args.smoke:
            keys=[feat.prepare(outputs[a][rows[0]["id"]],(0,10),"full") for a in ARMS]
            feat.run(keys,keys)
            assert all(np.isfinite(feat.cache[k]["clap"]).all() and np.isfinite(feat.cache[k]["panns"]).all()
                       for k in keys)
            write_json(root/"smoke_scores.json",dict(models=list(ARMS), per_arm_count=len(rows),
                                                     audio_headers_valid=True, clap_panns_finite=True))
            return

        models,distributions,calibration=score_formal(feat,rows,outputs,quality)
        result=dict(metrics={arm:model_summary(values) for arm,values in models.items()},
                    known_controls={name:model_summary(values) for name,values in calibration.items()},
                    paired=bootstrap(models), quality=distributions,
                    thresholds=dict(event_clap_margin=EVENT_CLAP_MIN,event_panns_margin=EVENT_PANNS_MIN,
                                    preservation_reference_margin=PRESERVATION_MIN, success_progress=PROGRESS_PASS),
                    detail=models,preference_unit="held-out source group",
                    native_aggregation="equal weight for available Add and Remove task means",
                    generated_quality_cut_seconds=10,
                    frechet_estimator="unregularized low-rank empirical covariance",
                    listening_status="pending independent listener ratings")
        write_json(root/"results.json",result)
        write_summary(root,result)


def score_formal(feat, rows, outputs, quality):
    by_id={r["id"]:r for r in rows}
    native={r["group"]:r for r in rows if r["kind"]=="add"}
    native_list=list(native.values())
    if len(native_list)<2:
        raise ValueError("Need distinct Add background references")
    for i,row in enumerate(native_list):
        for other in rows:
            if other["group"]==row["group"] and other["kind"] in {"add","remove"}:
                other["distractor_background"]=native_list[(i+1)%len(native_list)]["background"]
    texts=[]
    for r in rows:
        texts.extend(x for x in (description(r),r.get("source_category"),r.get("distractor_event"),
                                 r.get("target_description")) if x)
    feat.prime_text(texts)
    calibration={name:[] for name in ("source_copy","oracle")}
    models,distributions={},{}
    specs={}
    pending,panns_pending=[],[]

    def features(row, path, variants):
        keys={region:feat.prepare(path,case_span(row),region) for region in variants}
        pending.extend(keys.values())
        panns_pending.extend(key for region,key in keys.items() if region!="preserve")
        return keys

    for row in rows:
        variants=["full"]
        if row["kind"] in {"add","remove","swap_instruction","swap_source"}:
            variants += ["event","preserve"]
        source=features(row,row["source"],variants)
        target=features(row,row["target"],variants) if row["target"] else None
        bg=(features(row,row["background"],["preserve"])["preserve"] if row["background"] else None)
        alt=(features(row,row["distractor_background"],["preserve"])["preserve"]
             if row["distractor_background"] else None)
        specs[row["id"]]=(variants,source,target,bg,alt)
    feat.run(pending,panns_pending)
    retained=set(feat.cache)
    pending.clear()
    panns_pending.clear()

    for row in rows:
        kind=row["kind"]
        _,src,tgt,bg,alt=specs[row["id"]]
        if kind in {"swap_instruction","swap_source"}:
            old=by_id.get(row.get("parent_id")) or native[row["group"]]
            _,old_src,old_tgt,old_bg,_=specs[old["id"]]
            for name, first, second in (("source_copy",old_src,src),("oracle",old_tgt,tgt)):
                if kind=="swap_instruction":
                    cal=instruction_control(feat,first["full"],second["full"],old_tgt["full"],tgt["full"],
                                            description(old),description(row))
                else:
                    cal=source_control(feat,first["preserve"],second["preserve"],old_bg,bg)
                calibration[name].append(dict(id=row["id"],group=row["group"],kind=kind,**cal))
        if kind not in {"add","remove"}:
            continue
        for name,features_for,which in (("source_copy",src,"source"),("oracle",tgt,"target")):
            y,_=read_audio(row[which])
            cal=native_row(feat,row,y,features_for,tgt,src,bg,alt)
            calibration[name].append(dict(id=row["id"],group=row["group"],kind=row["kind"],**cal))

    for arm in ARMS:
        refs={}
        for row in rows:
            refs[row["id"]]=features(row,outputs[arm][row["id"]],specs[row["id"]][0])
        feat.run(pending,panns_pending)
        values=[]
        for row in rows:
            sid,kind=row["id"],row["kind"]
            _,src,tgt,bg,alt=specs[sid]
            got=refs[sid]
            y,_=read_audio(outputs[arm][sid],generated=True)
            stats=dict(id=sid,group=row["group"],kind=kind,**quality[arm][sid])
            if row.get("target_description"):
                stats["result_text_clap"]=feat.event(got["full"],row["target_description"])[0]
            if tgt:
                stats["target_audio_clap"]=feat.similarity(got["full"],tgt["full"])
            if kind in {"add","remove"}:
                stats.update(native_row(feat,row,y,got,tgt,src,bg,alt))
            elif kind=="attribute":
                original,_=read_audio(row["source"])
                oracle,_=read_audio(row["target"])
                stats["attribute_logmel_preference"]=log_mel_distance(y,original)-log_mel_distance(y,oracle)
                attr=str(row.get("attribute",row.get("attribute_type",""))).casefold()
                if any(token in attr for token in ("eq","low","bass")):
                    progress=relative_progress(lower_band_ratio(y),lower_band_ratio(original),
                                               lower_band_ratio(oracle),.5)
                    stats["attribute_band_progress"]=progress
                    stats["attribute_success"]=passed([progress])
            elif kind=="style":
                target_clap=feat.event(got["full"],row["target_category"])[0]
                source_clap=feat.event(got["full"],row["source_category"])[0]
                stats.update(style_target_clap=target_clap,style_text_margin=target_clap-source_clap)
                original,_=read_audio(row["source"])
                envelopes=[np.log(np.maximum(np.sqrt(np.mean(x.reshape(-1,2205)**2,axis=1)),1e-4))
                           for x in (original,y)]
                a,b=(np.diff(x) for x in envelopes)
                stats["onset_envelope_correlation"]=(float(np.corrcoef(a,b)[0,1])
                                                      if np.std(a)>0 and np.std(b)>0 else None)
            elif kind in {"swap_instruction","swap_source"}:
                old=by_id.get(row.get("parent_id")) or native[row["group"]]
                original=refs[old["id"]]
                _,_,original_target,old_bg,_=specs[old["id"]]
                if kind=="swap_instruction":
                    original_text, changed_text=description(old),description(row)
                    if not changed_text or original_text==changed_text:
                        raise ValueError(f"Instruction control is not modified: {sid}")
                    stats.update(instruction_control(feat,original["full"],got["full"],
                                                     original_target["full"],tgt["full"],
                                                     original_text,changed_text))
                else:
                    stats.update(source_control(feat,original["preserve"],got["preserve"],old_bg,bg))
            values.append(stats)
        models[arm]=values
        distributions[arm]={}
        for name, predicate in [
            ("native",lambda r:r["kind"] in {"add","remove"}),
            ("add",lambda r:r["kind"]=="add"),
            ("remove",lambda r:r["kind"]=="remove"),
            ("attribute",lambda r:r["kind"]=="attribute"),
        ]:
            ids=[r["id"] for r in rows if predicate(r) and r["target"]]
            if len(ids)>=10:
                distributions[arm][name]=dict(n=len(ids),**distribution(
                    [feat.cache[refs[i]["full"]] for i in ids],
                    [feat.cache[specs[i][2]["full"]] for i in ids]))
        for key in set(feat.cache)-retained:
            del feat.cache[key]
            feat.paths[key].unlink()
            # Keep the path key to guarantee monotonically assigned temp filenames.
        pending.clear()
        panns_pending.clear()
        print(f"scored {arm}: {len(values)} cases",flush=True)
    return models,distributions,calibration


def write_summary(root, result):
    def fmt(metric):
        return f"{100*metric['mean']:.1f}% ({metric['n']})" if metric else "—"
    lines=["# Audio Editing 迁移评测", "",
           "指标为机器代理。百分率后的括号给出有可校准证据的样本数。"
           "独立盲听评分待收集。配对区间使用源组重采样。", "",
           "| 模型 | 编辑成功 | 保留成功 | 同时成功 | 只完成编辑 | 只保留内容 | 都未达到 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, payload in result["metrics"].items():
        s=payload["groups"]["native"]
        c=payload["native_contingency_table"]
        fields=[fmt(s.get(k)) for k in ("edit_success","preserve_success","joint_success")]
        fields += [str(c[k]) for k in ("edit_only","preserve_only","neither")]
        lines.append("| "+" | ".join([arm,*fields])+" |")
    lines+=["","| 对照 | 编辑成功率差 | 保留成功率差 | 联合成功率差 | 换指令成功率差 | 换源成功率差 |",
            "|---|---:|---:|---:|---:|---:|"]
    def contrast_format(data):
        if data is None:
            return "—"
        lo,hi=data["ci95"]
        return f"{data['delta']:+.1%} [{lo:+.1%}, {hi:+.1%}]"
    for name, row in result["paired"].items():
        values=[row["native"][k] for k in ("edit_success","preserve_success","joint_success")]
        values+=[row["swap_instruction"]["instruction_switch_success"],row["swap_source"]["source_switch_success"]]
        lines.append("| "+" | ".join([name.replace("__minus__"," − "),*[contrast_format(v) for v in values]])+" |")
    lines += ["", "## 预定重点对照", ""]
    comparisons = {
        "pure_7070__minus__base": "Pure ReCA 相对 Base",
        "mix_7070__minus__text_slots_7070": "Mix 相对同 caption 预算仅文本槽",
        "mix_pool4_7070__minus__mix_7070": "Pool4 相对原 Mix",
    }
    def direction(metric):
        if metric is None:
            return "缺少可校准样本"
        lo,hi=metric["ci95"]
        if lo>0:
            return "提高"
        if hi<0:
            return "下降"
        return "区间跨零"
    for name, title in comparisons.items():
        row=result["paired"][name]
        change=[(label,direction(row["native"][k]))
                for label,k in (("编辑成功","edit_success"),("内容保留","preserve_success"),
                                ("联合成功","joint_success"))]
        change.extend((label,direction(row[kind][key])) for label,kind,key in
                      (("换指令","swap_instruction","instruction_switch_success"),
                       ("换源","swap_source","source_switch_success")))
        lines.append(f"- {title}，"+"，".join(label+" "+value for label,value in change)+"。")
    lines += ["", "把联合成功与换指令的结果一起解读。只有内容保留或换源指标改善时，"
              "证据限于对输入源的响应。质量指标为嵌入分布代理，听感结论以盲评为准。"]
    lines+=["","| 波形对照 | 编辑成功 | 保留成功 | 同时成功 |","|---|---:|---:|---:|"]
    for name, payload in result["known_controls"].items():
        s=payload["groups"]["native"]
        vals=[fmt(s.get(k)) for k in ("edit_success","preserve_success","joint_success")]
        lines.append("| "+" | ".join([name,*vals])+" |")
    lines+=["","| 模型 | CLAP-FAD ↓ | PANNs FD ↓ | KL ↓ | IS ↑ |","|---|---:|---:|---:|---:|"]
    for arm, distributions in result["quality"].items():
        x=distributions["native"]
        lines.append(f"| {arm} | {x['clap_fad']:.5f} | {x['panns_fd']:.3f} | {x['panns_kl']:.4f} | {x['panns_is']:.4f} |")
    lines+=["","完整分布指标、按任务统计、校准对照和逐条配对分数见 [results.json](results.json)。"]
    (root/"analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
