"""scripts/run_all.py: the machine-dependent sizing and the CUDA wheel choice."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "run_all", Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
)
run_all = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_all)


@pytest.mark.parametrize(
    ("cuda", "tag"),
    [("13.1", "cu130"), ("12.9", "cu128"), ("12.8", "cu128"), ("12.6", "cu126"),
     ("12.4", "cu124"), ("12.2", "cu121"), ("11.8", "cu118")],
)  # fmt: skip
def test_torch_wheel_follows_the_driver(cuda, tag):
    assert run_all.torch_wheel_index(cuda).endswith(tag)


def test_old_driver_is_refused():
    with pytest.raises(SystemExit):
        run_all.torch_wheel_index("11.2")


def _hw(free_gb, ram_gb, cpus=16):
    return {"gpu": {"free_gb": free_gb}, "ram_total_gb": ram_gb, "cpus": cpus, "bf16": True}


def test_profile_grows_with_the_machine():
    small = run_all.make_profile(_hw(3.5, 16, 8), smoke=False)
    big = run_all.make_profile(_hw(22, 64, 32), smoke=False)
    for key in ("embed_batch", "ce_batch", "score_batch", "n_threads", "train_sample"):
        assert big[key] > small[key], key
    assert small["lw_grad_ckpt"] and not big["lw_grad_ckpt"]
    assert big["large_ok"] and not small["large_ok"]
    assert small["block_chunk"] < big["block_chunk"]
    assert all(small[k] >= 4 for k in ("lw_batch", "ce_batch", "embed_batch"))
    assert big["embed_batch"] <= 1024 and big["ce_batch"] <= 512 and big["lw_batch"] <= 128


def test_smoke_profile_is_tiny():
    p = run_all.make_profile(_hw(16, 32), smoke=True)
    assert p["train_sample"] <= 1000 and p["ce_max_pairs"] <= 50_000 and p["lw_train"] <= 1000


def test_memory_probe_works_here():
    total, avail = run_all.ram_gb()
    assert total > 0 and 0 <= avail <= total * 1.01
