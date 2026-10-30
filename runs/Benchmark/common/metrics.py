"""评测指标。

默认实现（不需要额外重权重）：
- FAD / FD：CLAP 音频嵌入上的 Fréchet 距离（`--embedding clap`）
- CLAP score：文本-音频余弦相似度
- LSD：对数谱距离

可选实现（装了才用，否则跳过并给出提示）：
- VGGish-FAD：`pip install fadtk` 后可用，数值才与论文可比
- KL / IS / FD(PANNs)：`pip install panns-inference` 后可用，对应论文的 PANNs 协议
- WER：transformers 的 Whisper pipeline

注意：CLAP-FAD 与论文的 VGGish-FAD **不可直接比较**，只适合同协议下的横向比较。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# 基础
# ----------------------------------------------------------------------
def _as_files(paths: Sequence[str]) -> List[str]:
    return [str(p) for p in paths]


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    """两个高斯分布之间的 Fréchet 距离。样本数太少时返回 nan。"""
    from scipy import linalg

    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    if a.shape[0] < 10 or b.shape[0] < 10:
        return float("nan")
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * eps
    cov_b = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * eps
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean))


# ----------------------------------------------------------------------
# CLAP 嵌入
# ----------------------------------------------------------------------
_CLAP = None


def get_clap(device: str = "cuda"):
    global _CLAP
    if _CLAP is None:
        import laion_clap

        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        model.load_ckpt()
        model.eval()
        _CLAP = model
    return _CLAP


def clap_audio_embeddings(files: Sequence[str], device: str = "cuda") -> np.ndarray:
    model = get_clap(device)
    emb = model.get_audio_embedding_from_filelist(x=_as_files(files), use_tensor=False)
    return np.asarray(emb, dtype=np.float64)


def clap_text_audio_score(files: Sequence[str], texts: Sequence[str], device: str = "cuda") -> float:
    model = get_clap(device)
    audio = np.asarray(model.get_audio_embedding_from_filelist(x=_as_files(files), use_tensor=False))
    text = np.asarray(model.get_text_embedding(list(texts), use_tensor=False))
    audio = audio / (np.linalg.norm(audio, axis=1, keepdims=True) + 1e-8)
    text = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(audio * text, axis=1)))


# ----------------------------------------------------------------------
# FAD / FD
# ----------------------------------------------------------------------
def fad_clap(gen_files: Sequence[str], ref_files: Sequence[str], device: str = "cuda") -> float:
    """CLAP 嵌入上的 Fréchet 距离。"""
    return frechet_distance(clap_audio_embeddings(gen_files, device), clap_audio_embeddings(ref_files, device))


def fad_vggish(gen_files: Sequence[str], ref_files: Sequence[str], sr: int = 16000) -> float:
    """需要 fadtk；否则返回 nan。"""
    try:
        from fadtk import FrechetAudioDistance
        from fadtk.model_loader import VggishModel
    except Exception:
        print("[metrics] 未安装 fadtk，跳过 VGGish-FAD（pip install fadtk）")
        return float("nan")
    try:
        fad = FrechetAudioDistance(VggishModel())
        return float(fad.score(_as_files(ref_files), _as_files(gen_files)))
    except Exception as e:  # noqa: BLE001
        print(f"[metrics] VGGish-FAD 计算失败：{e}")
        return float("nan")


# ----------------------------------------------------------------------
# KL / IS / FD（PANNs 协议，可选）
# ----------------------------------------------------------------------
_PANNS = None


def get_panns(device: str = "cuda"):
    global _PANNS
    if _PANNS is None:
        from panns_inference import AudioTagging

        _PANNS = AudioTagging(checkpoint_path=None, device=device)
    return _PANNS


def panns_outputs(files: Sequence[str], device: str = "cuda"):
    import torch

    model = get_panns(device)
    logits, embeds = [], []
    for path in _as_files(files):
        wav, sr = _load_fixed_32k(path)
        with torch.no_grad():
            clipwise, embedding = model.inference(wav[None, :])
        logits.append(np.asarray(clipwise).reshape(-1))
        embeds.append(np.asarray(embedding).reshape(-1))
    return np.stack(logits), np.stack(embeds)


def _load_fixed_32k(path: str) -> np.ndarray:
    from .audio_io import load_audio

    wav, _ = load_audio(path, sr=32000, mono=True)
    return wav.squeeze(0).numpy()


def panns_scores(gen_files: Sequence[str], ref_files: Sequence[str], device: str = "cuda") -> Dict[str, float]:
    """KL / IS / FD，对应论文的 PANNs 协议。未装 panns_inference 时返回空。"""
    try:
        gen_logits, gen_emb = panns_outputs(gen_files, device)
        ref_logits, ref_emb = panns_outputs(ref_files, device)
    except Exception as e:  # noqa: BLE001
        print(f"[metrics] PANNs 不可用，跳过 KL/IS/FD：{e}")
        return {}

    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    p_gen, p_ref = softmax(gen_logits), softmax(ref_logits)
    p_ref_mean = p_ref.mean(axis=0, keepdims=True)
    eps = 1e-10
    kl_gen_ref = float(np.mean(np.sum(p_gen * np.log((p_gen + eps) / (p_ref_mean + eps)), axis=1)))
    kl_self = np.sum(p_gen * np.log((p_gen + eps) / (p_gen.mean(axis=0, keepdims=True) + eps)), axis=1)
    return {
        "kl": kl_gen_ref,
        "is": float(np.exp(np.mean(kl_self))),
        "fd": frechet_distance(gen_emb, ref_emb),
    }


# ----------------------------------------------------------------------
# LSD
# ----------------------------------------------------------------------
def _stft_mag(path: str, n_fft: int = 1024, hop: int = 256, sr: int = 44100) -> np.ndarray:
    from .audio_io import load_audio

    wav, _ = load_audio(path, sr=sr, mono=True)
    import torch

    window = torch.hann_window(n_fft)
    spec = torch.stft(wav.squeeze(0), n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    return spec.abs().numpy()


def lsd(gen_files: Sequence[str], ref_files: Sequence[str], n_fft: int = 1024, hop: int = 256, sr: int = 44100) -> float:
    """对数谱距离（dB），按较短长度对齐。"""
    vals = []
    for g, r in zip(gen_files, ref_files):
        mg, mr = _stft_mag(g, n_fft, hop, sr), _stft_mag(r, n_fft, hop, sr)
        n = min(mg.shape[1], mr.shape[1])
        if n == 0:
            continue
        diff = 20.0 * np.log10((mg[:, :n] + 1e-8) / (mr[:, :n] + 1e-8))
        vals.append(float(np.sqrt(np.mean(diff**2))))
    return float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------------
# WER
# ----------------------------------------------------------------------
def wer(gen_files: Sequence[str], ref_texts: Sequence[str], model_name: str = "openai/whisper-small", device: str = "cuda") -> float:
    """用 Whisper 转写生成音频，与参考文本算 WER。"""
    import jiwer
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model_name, device=device)
    hyps = []
    for path in gen_files:
        out = asr(str(path), generate_kwargs={"language": "en"})
        hyps.append(out["text"].strip())
    return float(jiwer.wer(list(ref_texts), hyps))
