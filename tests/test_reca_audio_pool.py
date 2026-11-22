"""Exercise the actual conditioner with deterministic Qwen outputs on CPU."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def conditioner(monkeypatch):
    # These imports are only used by the unrelated TTS conditioner.
    for name, stub in (("jieba", SimpleNamespace()), ("torchaudio", SimpleNamespace()),
                       ("pypinyin", SimpleNamespace(Style=None, lazy_pinyin=None))):
        monkeypatch.setitem(sys.modules, name, stub)
    path = Path(__file__).resolve().parents[1] / "Audio-Omni/audio_omni/models/conditioners.py"
    spec = importlib.util.spec_from_file_location("pool_test_conditioners", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Processor:
        tokenizer = SimpleNamespace(convert_tokens_to_ids={"<|AUDIO|>": 42}.get,
                                    convert_ids_to_tokens={42: "<|AUDIO|>"}.get)
        # Left padding; runs of 6, 2 and 1 payload tokens, separated by a
        # masked gap and an instruction token. The second sample is text only.
        ids = torch.tensor([[0, 9, 42, 42, 42, 42, 42, 42, 0, 42, 42, 7, 42, 6],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 6]])

        def __call__(self, **kwargs):
            return {"input_ids": self.ids, "attention_mask": self.ids.ne(0)}

    class Thinker(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            pos = torch.arange(input_ids.shape[1]).expand_as(input_ids)
            hidden = torch.stack((input_ids, pos), dim=-1).float()
            return SimpleNamespace(hidden_states=(hidden,))

    net = module.OmniConditioner.__new__(module.OmniConditioner)
    torch.nn.Module.__init__(net)
    net.processor = Processor()
    net.model = SimpleNamespace(thinker=Thinker())
    net.pad_length = 16
    net.layer_idx = 0
    net._omni_dtype = torch.float32
    net.norm = module.RMSNorm(2)
    net.proj_features = torch.nn.Linear(2, 3)
    return net


def test_pooling_uses_projected_audio_only_and_keeps_partial_windows(conditioner):
    audio = dict(text_prompt="audio", audio_prompt=[1., -1.])
    text = dict(text_prompt="caption", audio_prompt=None)
    raw, mask = conditioner([audio, text], "cpu")
    pooled, new_mask = conditioner([dict(audio, reca_audio_pool_width=4), text], "cpu")
    expected = raw.clone()
    for start, end in ((1, 5), (5, 7), (7, 9), (10, 11)):
        expected[0, start:end] = raw[0, start:end].mean(dim=0)
    assert torch.equal(mask, new_mask) and not torch.equal(raw, pooled)
    torch.testing.assert_close(pooled, expected)
    assert torch.equal(raw[1], pooled[1])
    assert torch.equal(raw[0, [0, 9, 11]], pooled[0, [0, 9, 11]])
    assert torch.equal(raw[~mask], pooled[~mask])


def test_rejects_unmappable_or_missing_payload(conditioner):
    prompts = [dict(text_prompt="audio", audio_prompt=[1.], reca_audio_pool_width=4),
               dict(text_prompt="caption")]
    conditioner.pad_length = 10
    with pytest.raises(ValueError, match="interpolation"):
        conditioner(prompts, "cpu")
    conditioner.pad_length = 16
    conditioner.zip_features = True
    with pytest.raises(ValueError, match="unzipped"):
        conditioner(prompts, "cpu")
    conditioner.zip_features = False
    conditioner.processor.ids = conditioner.processor.ids.masked_fill(conditioner.processor.ids.eq(42), 9)
    with pytest.raises(ValueError, match="no valid audio"):
        conditioner(prompts, "cpu")
