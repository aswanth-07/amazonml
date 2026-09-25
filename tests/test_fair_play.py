"""Guards for the challenge's fair-play and model-licence rules."""

import ast
import sys
from pathlib import Path

import pytest

from ber.pretrained import (
    ALLOWED_LICENSES,
    MAX_PARAMETERS,
    REGISTRY,
    DisallowedModelError,
    check_allowed,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "ber"

# External lookups of business identities, registries or geocoders mean
# disqualification. Rather than list what is forbidden (an open-ended set), the
# pipeline may import only the standard library, minus its networking modules,
# and the declared dependencies below. A new third-party import fails here
# until someone adds it on purpose.
THIRD_PARTY = {
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
    "rapidfuzz",
    "anyascii",
    "yaml",
    "joblib",
}
STDLIB_NETWORK = {
    "socket", "ssl", "http", "urllib", "ftplib", "smtplib", "poplib", "imaplib", "nntplib",
    "telnetlib", "xmlrpc", "asyncio", "webbrowser", "socketserver",
}  # fmt: skip
# Sanctioned exceptions: loading and fetching licence-checked encoder weights
# for the optional, offline embedding blocker.
EXCEPTIONS = {
    ("pretrained.py", "huggingface_hub"),
    ("blockers.py", "sentence_transformers"),
    # fine-tuning and running the local encoder (weights on disk, no network)
    ("embed.py", "torch"),
    ("embed.py", "transformers"),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "path", sorted(SRC.rglob("*.py")), ids=lambda p: p.relative_to(SRC).as_posix()
)
def test_imports_are_offline_stdlib_or_declared(path: Path):
    for name in _imports(path):
        top = name.split(".")[0]
        if top == "ber" or (path.name, top) in EXCEPTIONS:
            continue
        if top in sys.stdlib_module_names:
            assert top not in STDLIB_NETWORK, f"{path.name} imports networking module {name}"
        else:
            assert top in THIRD_PARTY, f"{path.name} imports undeclared third-party module {name}"


def test_allowlist_matches_declared_dependencies():
    """Every allowed third-party import is a pinned runtime requirement."""
    pinned = (SRC.parents[1] / "requirements.txt").read_text(encoding="utf-8").lower()
    dist_names = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
    for module in THIRD_PARTY:
        assert f"{dist_names.get(module, module)}==" in pinned, module


def test_registry_respects_licence_and_size():
    for name, info in REGISTRY.items():
        assert info.license in ALLOWED_LICENSES, name
        assert info.parameters <= MAX_PARAMETERS, name
        check_allowed(name)


def test_unregistered_model_is_refused():
    with pytest.raises(DisallowedModelError):
        check_allowed("some-org/llama-70b")


def test_no_hard_coded_country_filter():
    """Country labels must stay an open set: nothing may filter to {US, India}."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "synth.py":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in (
            '== "US"',
            "== 'US'",
            '== "India"',
            "== 'India'",
            'isin(["US"',
            "isin(['US'",
        ):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders
