"""ReCA 后训练：把 Audio-Omni 的生成头对齐到“理解塔的表征”。

模块划分：
- data.py          音频数据集（manifest jsonl）
- conditioning.py  ReCA / T2A 两种条件构造
- flow.py          rectified flow 损失
- train.py         训练主循环
"""

from .data import AudioManifestDataset, collate_clips  # noqa: F401
from .conditioning import build_reca_metadata, build_t2a_metadata  # noqa: F401
from .flow import rectified_flow_loss  # noqa: F401

__all__ = [
    "AudioManifestDataset",
    "collate_clips",
    "build_reca_metadata",
    "build_t2a_metadata",
    "rectified_flow_loss",
]
