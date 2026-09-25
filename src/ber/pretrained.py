"""Registry of pretrained models the rules allow, and the one place that downloads.

Challenge rule: the final model must be MIT or Apache-2.0 licensed with at most
8B parameters. Anything not listed here is refused before it is loaded. Adding
an entry means checking the model card's licence yourself first.

Downloading weights is a setup step (``ber fetch-model``), never part of the
pipeline run, and it fetches a model, not data about any business.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ALLOWED_LICENSES = frozenset({"MIT", "Apache-2.0"})
MAX_PARAMETERS = 8_000_000_000


@dataclass(frozen=True)
class ModelInfo:
    license: str
    parameters: int  # approximate, from the model card
    note: str = ""


REGISTRY: dict[str, ModelInfo] = {
    "intfloat/multilingual-e5-small": ModelInfo("MIT", 118_000_000, "multilingual; handles French"),
    "intfloat/multilingual-e5-base": ModelInfo("MIT", 278_000_000, "multilingual"),
    "BAAI/bge-small-en-v1.5": ModelInfo("MIT", 33_000_000, "English only"),
    "BAAI/bge-m3": ModelInfo("MIT", 568_000_000, "multilingual, 8k context"),
    "BAAI/bge-reranker-v2-m3": ModelInfo("Apache-2.0", 568_000_000, "multilingual reranker"),
    "sentence-transformers/all-MiniLM-L6-v2": ModelInfo("Apache-2.0", 23_000_000, "English only"),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ModelInfo(
        "Apache-2.0", 118_000_000, "multilingual"
    ),
}


class DisallowedModelError(ValueError):
    pass


def check_allowed(name: str) -> ModelInfo:
    info = REGISTRY.get(name)
    if info is None:
        raise DisallowedModelError(
            f"{name!r} is not in ber.pretrained.REGISTRY; verify its licence (MIT/Apache-2.0) "
            "and size (<= 8B) on the model card, then register it"
        )
    if info.license not in ALLOWED_LICENSES or info.parameters > MAX_PARAMETERS:
        raise DisallowedModelError(f"{name!r} violates the licence/size rule: {info}")
    return info


def fetch(name: str, dest: str | Path) -> Path:  # pragma: no cover - network
    check_allowed(name)
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=name, local_dir=str(dest))
    return Path(path)
