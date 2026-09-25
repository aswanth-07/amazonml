"""The whole v6 run in one command, sized to the machine it runs on.

    python scripts/run_all.py --data-dir PATH/TO/dataset          # everything, resumable
    python scripts/run_all.py --data-dir PATH --only hardware     # just print the profile
    python scripts/run_all.py --data-dir PATH --from runC         # redo from a step on

It starts with the standard library only, so it can install everything else itself
(``setup``: the pipeline's pinned requirements, PyTorch built for this machine's CUDA
driver, transformers). Then it measures the machine (GPU and its free memory, bfloat16
support, CPU threads, RAM, disk) and sizes every batch, thread count, chunk and training
sample from what it found. Each step logs to ``runs/v6/logs/<step>.log`` and leaves a
marker in ``runs/v6/done/``, so a rerun resumes after the last finished step. A GPU step
that runs out of memory is retried with half the batch.

Steps (v5's recipe from ``runs/v5_overnight.sh``, from scratch, plus v6's new models):

 1. setup       install requirements and CUDA PyTorch into this interpreter
 2. hardware    measure the machine, write runs/v6/profile.json
 3. models      download the base encoder multilingual-e5-small (MIT)
 4. embed       fine-tune the bi-encoder, encode train + test, kNN lists        (GPU)
 5. runA        blocking, features, stage 1 + 2 without cross-encoders          (CPU) -> output-A
 6. ce          pairwise cross-encoder on the encoder share                     (GPU)
 7. evalpairs   held-out pairs that gate adapted cross-encoders
 8. runB        + cross-encoder, pool-side context (~ v4)                       (CPU) -> output-B
 9. pseudo      French pseudo-labels where runs A and B agree confidently
10. cefr        France-adapted cross-encoder, promoted only if it holds on US/India (GPU)
11. groups      C1 training groups from the raw data (encoder share)            (CPU)
12. pgroups     C7 training groups (pool side)                                   (CPU)
13. lw          C1 listwise reranker                                             (GPU)
14. lwp         C7 pool-side listwise reranker                                   (GPU)
15. lweval      C1 / C7 against stage 1 and the cross-encoder on held-out groups (GPU)
16. runC        v6: France CE + C1 + C7 + source fingerprints, stage-2 inputs saved -> output-C
17. search      every stage-2 model family on the holdout, with and without the listwise columns
18. final       the best combination (scores cached)                             -> output-final
19. report      runs/v6/RESULTS.md: every holdout number and what to submit

``--large`` adds a France-adapted bge-reranker-v2-m3 (568M, Apache-2.0) between steps 10
and 11 when the GPU has >= 14 GB free (about 3 extra hours).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "runs" / "v6"
PY = sys.executable
OOM_MARKERS = (
    "out of gpu memory",
    "outofmemoryerror",
    "cuda out of memory",
    "cuda error: out of memory",
)
STEPS = ["setup", "hardware", "models", "embed", "runA", "ce", "evalpairs", "runB", "pseudo",
         "cefr", "large", "groups", "pgroups", "lw", "lwp", "lweval", "runC", "search", "final",
         "report"]  # fmt: skip
GPU_PINS = ["transformers==5.5.4", "tokenizers==0.22.2", "safetensors==0.7.0",
            "huggingface_hub==1.11.0"]  # fmt: skip


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- state
def load_state() -> dict:
    p = WORK / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(state: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "state.json").write_text(json.dumps(state, indent=2))


def done(step: str) -> bool:
    return (WORK / "done" / step).exists()


def mark(step: str) -> None:
    (WORK / "done").mkdir(parents=True, exist_ok=True)
    (WORK / "done" / step).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# ----------------------------------------------------------------------------- processes
def run(cmd: list[str], step: str, env: dict | None = None, oom_limit: int = 25) -> str:
    """Run ``cmd`` with output to runs/v6/logs/<step>.log; print its latest line every
    minute. Returns "ok", "oom" (killed after ``oom_limit`` out-of-memory reports, or
    crashed with one) or raises on any other failure."""
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{step}.log"
    full_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT / "src"),
        **(env or {}),
    }
    say(f"{step}: {' '.join(str(c) for c in cmd)}")
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        log.flush()
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=full_env, stdout=log,
                                stderr=subprocess.STDOUT)  # fmt: skip
        t0, last_print, offset, ooms = time.time(), 0.0, log_path.stat().st_size, 0
        while proc.poll() is None:
            time.sleep(5)
            with open(log_path, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                new = f.read()
                offset = f.tell()
            lines = [ln for ln in new.splitlines() if ln.strip()]
            ooms += sum(any(m in ln.lower() for m in OOM_MARKERS) for ln in lines)
            if ooms >= oom_limit:
                proc.kill()
                proc.wait()
                say(f"{step}: {ooms} out-of-memory reports; stopped to retry with a smaller batch")
                return "oom"
            if lines and time.time() - last_print > 60:
                say(f"{step} [{(time.time() - t0) / 60:.0f} min] {lines[-1][:150]}")
                last_print = time.time()
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if proc.returncode != 0:
        if any(m in tail.lower() for m in OOM_MARKERS):
            return "oom"
        print(tail[-2500:])
        raise SystemExit(f"{step} failed (exit {proc.returncode}); full log: {log_path}")
    say(f"{step}: finished in {(time.time() - t0) / 60:.1f} min")
    return "ok"


def run_gpu(make_cmd, step: str, batch: int, env: dict | None = None, floor: int = 4) -> int:
    """Run a GPU step, halving its batch after an out-of-memory stop (at most 4 times).
    Returns the batch that worked."""
    for _ in range(5):
        if run(make_cmd(batch), step, env) == "ok":
            return batch
        if batch <= floor:
            break
        batch = max(floor, batch // 2)
        say(f"{step}: retrying with batch {batch}")
    raise SystemExit(
        f"{step}: out of GPU memory even at batch {batch}; see runs/v6/logs/{step}.log"
    )


def latest_run(tag: str) -> Path:
    runs = sorted((ROOT / "runs").glob(f"*-{tag}"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit(f"no run tagged {tag} under runs/")
    return runs[-1]


# ----------------------------------------------------------------------------- hardware
def ram_gb() -> tuple[float, float]:
    """(total, available) GB, standard library only."""
    system = platform.system()
    if system == "Windows":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]  # fmt: skip

        m = MemoryStatus()
        m.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
    if Path("/proc/meminfo").exists():
        info = dict(re.findall(r"(\w+):\s+(\d+) kB", Path("/proc/meminfo").read_text()))
        return int(info["MemTotal"]) / 2**20, int(info.get("MemAvailable", info["MemFree"])) / 2**20
    total = int(
        subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
    )
    return total / 2**30, total / 2**30 / 2


def gpu_info() -> dict:
    """First NVIDIA GPU from nvidia-smi: name, total and free memory (GB), driver, CUDA."""
    if not shutil.which("nvidia-smi"):
        return {}
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)  # fmt: skip
    if q.returncode != 0 or not q.stdout.strip():
        return {}
    name, total, used, driver = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
    head = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", head)
    return {"name": name, "total_gb": round(int(total) / 1024, 1),
            "free_gb": round((int(total) - int(used)) / 1024, 1), "driver": driver,
            "cuda": cuda.group(1) if cuda else ""}  # fmt: skip


def torch_wheel_index(cuda: str) -> str:
    """The PyTorch wheel index for a driver's highest supported CUDA version."""
    v = tuple(int(x) for x in (cuda or "0").split(".")[:2])
    for need, tag in (((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"),
                      ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")):  # fmt: skip
        if v >= need:
            return f"https://download.pytorch.org/whl/{tag}"
    raise SystemExit(
        f"NVIDIA driver reports CUDA {cuda or '?'}; update the driver (>= 11.8 needed)"
    )


def make_profile(hw: dict, smoke: bool) -> dict:
    """Every size the run uses, from the measured machine."""
    free = hw["gpu"].get("free_gb", 0.0)
    ram_total = hw["ram_total_gb"]
    cpus = hw["cpus"]
    p = {
        "embed_batch": int(min(1024, max(64, free * 45)) // 64 * 64),
        "ce_batch": int(min(512, max(32, free * 28)) // 16 * 16),
        "lw_batch": int(min(64, max(4, free * 2.2)) // 4 * 4),
        "lw_grad_ckpt": free < 8,
        "score_batch": int(min(256, max(32, free * 12)) // 16 * 16),
        "n_threads": max(2, cpus - 2),
        "n_jobs": max(1, min(cpus // 2, int(ram_total // 3), 12)),
        "train_sample": 700_000 if ram_total >= 60 else 500_000 if ram_total >= 28 else 300_000,
        "block_chunk": 200_000 if ram_total >= 24 else 100_000,
        "feature_chunk": 2_000_000 if ram_total >= 24 else 1_000_000,
        "large_ok": free >= 14,
        "amp": "bf16" if hw.get("bf16") else "fp16",
    }
    if p["lw_grad_ckpt"]:
        p["lw_batch"] = max(4, p["lw_batch"] * 2)
    p.update({"ce_max_pairs": 4_000_000, "ce_fr_pairs": 1_200_000, "lw_train": 400_000,
              "lwp_train": 800_000, "pseudo_pos": 600_000, "pseudo_neg": 600_000,
              "eval_entities": 60_000, "search_sample": 250_000, "embed_epochs": 0.5,
              "lw_epochs": 1.0})  # fmt: skip
    if smoke:  # synthetic data: tiny everything, same code paths
        p.update({"train_sample": 250, "ce_max_pairs": 20_000, "ce_fr_pairs": 10_000,
                  "lw_train": 300, "lwp_train": 600, "pseudo_pos": 5_000, "pseudo_neg": 5_000,
                  "eval_entities": 100, "search_sample": 0, "embed_epochs": 0.3,
                  "embed_batch": min(p["embed_batch"], 64), "lw_epochs": 1.0,
                  "block_chunk": 200_000, "feature_chunk": 2_000_000})  # fmt: skip
    return p


# ----------------------------------------------------------------------------- steps
def step_setup(args, state) -> None:
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit(f"Python {platform.python_version()}: use Python 3.10-3.13")
    if sys.prefix == sys.base_prefix:
        say("note: not in a virtual environment; packages go into this Python")
    pip = [PY, "-m", "pip", "install", "--upgrade"]
    run([*pip, "pip"], "setup")
    run([*pip, "-r", "requirements-dev.txt"], "setup")
    run([PY, "-m", "pip", "install", "-e", ".", "--no-deps"], "setup")
    probe = "import torch;print(int(torch.cuda.is_available()))"
    have = subprocess.run([PY, "-c", probe], capture_output=True, text=True)
    if have.returncode != 0 or have.stdout.strip() != "1":
        gpu = gpu_info()
        if not gpu:
            raise SystemExit("no NVIDIA GPU found (nvidia-smi); the GPU steps need one")
        run([*pip, "torch", "--index-url", torch_wheel_index(gpu["cuda"])], "setup")
    run([*pip, *GPU_PINS], "setup")
    check = subprocess.run([PY, "-c", "import torch;print(torch.__version__, torch.cuda.is_available(),"
                            " torch.cuda.get_device_name(0))"], capture_output=True, text=True)  # fmt: skip
    say(f"setup: torch {check.stdout.strip()}")
    if "True" not in check.stdout:
        raise SystemExit("PyTorch cannot see the GPU; check the NVIDIA driver")


def step_hardware(args, state) -> None:
    total, avail = ram_gb()
    gpu = gpu_info()
    probe = subprocess.run([PY, "-c", "import torch;print(torch.cuda.is_bf16_supported(),"
                            "torch.cuda.mem_get_info()[0]/2**30)"], capture_output=True, text=True)  # fmt: skip
    bf16, free = False, gpu.get("free_gb", 0.0)
    if probe.returncode == 0 and probe.stdout.strip():
        a, b = probe.stdout.split()
        bf16, free = a == "True", round(float(b), 1)
    gpu["free_gb"] = free
    hw = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
          "cpus": os.cpu_count() or 4, "ram_total_gb": round(total, 1), "ram_free_gb": round(avail, 1),
          "disk_free_gb": round(shutil.disk_usage(ROOT).free / 2**30, 1), "gpu": gpu, "bf16": bf16}  # fmt: skip
    prof = make_profile(hw, args.smoke)
    state.update({"hardware": hw, "profile": prof})
    (WORK / "profile.json").write_text(json.dumps({"hardware": hw, "profile": prof}, indent=2))
    say(f"hardware: {json.dumps(hw)}")
    say(f"profile: {json.dumps(prof)}")
    if not gpu:
        raise SystemExit("no NVIDIA GPU visible")
    if hw["ram_total_gb"] < 24:
        say("WARNING: under 24 GB RAM; smaller chunks and sample chosen, close other apps")
    if hw["disk_free_gb"] < 150 and not args.smoke:
        say(f"WARNING: {hw['disk_free_gb']} GB free disk; the caches need ~120-150 GB")


def step_models(args, state) -> None:
    if not (ROOT / "models/multilingual-e5-small/config.json").exists():
        run([PY, "-m", "ber.cli", "fetch-model", "intfloat/multilingual-e5-small", "--dest",
             "models/multilingual-e5-small"], "models")  # fmt: skip


def step_embed(args, state) -> None:
    p = state["profile"]
    run_gpu(lambda b: [PY, "-m", "ber.embed", "all", "--data-dir", args.data_dir,
                       "--base", "models/multilingual-e5-small", "--model", "models/e5-small-ft",
                       "--emb-dir", "artifacts/emb", "--epochs", str(p["embed_epochs"]),
                       "--batch", str(b), "--k", "50", "--k-reverse", "3"],
            "embed", p["embed_batch"], floor=32)  # fmt: skip


def common_sets(args, state) -> list[str]:
    """Settings every pipeline run shares, so stage 1 is reusable between them."""
    p = state["profile"]
    ce_python = f"scale.ce_python='\"{Path(PY).as_posix()}\"'"
    sets = ["scale.emb_dir=artifacts/emb", "scale.k_emb=20", f"scale.train_sample={p['train_sample']}",
            f"scale.drop_s1_share={args.drop}", f"scale.n_threads={p['n_threads']}",
            f"scale.n_jobs={p['n_jobs']}", f"scale.block_chunk={p['block_chunk']}",
            f"scale.feature_chunk={p['feature_chunk']}", ce_python,
            "scale.ce_pool_context=true", "scale.ce_train_all=true"]  # fmt: skip
    if args.smoke:
        sets += ["model.n_folds=3", "model.num_boost_round=100", "model.early_stopping_rounds=20",
                 "scale.ce_min_p1=0.001"]  # fmt: skip
    out = []
    for s in sets:
        out += ["--set", s]
    return out


def pipeline(args, state, tag: str, extra: list[str]) -> Path:
    cmd = [PY, "-m", "ber.cli", "run", "--config", "src/ber/configs/full.yaml",
           "--data-dir", args.data_dir, "--out-dir", f"output-{tag}", "--tag", tag,
           *common_sets(args, state)]  # fmt: skip
    for s in extra:
        cmd += ["--set", s]
    env = {"BER_LISTWISE_SCORE_BATCH": str(state["profile"]["score_batch"])}
    run(cmd, f"run{tag}" if tag in ("A", "B", "C") else tag, env)
    run_dir = latest_run(tag)
    state[f"run_{tag}"] = str(run_dir)
    return run_dir


def step_runA(args, state) -> None:
    pipeline(args, state, "A", [])


def blocking_cache() -> str:
    caches = sorted((ROOT / "artifacts/cache").glob("train_blocking_*.joblib"),
                    key=lambda p: p.stat().st_mtime)  # fmt: skip
    if not caches:
        raise SystemExit("no train blocking cache; run runA first")
    return str(caches[-1].relative_to(ROOT))


def ce_train_cmd(args, init: str, out: str, pairs: int, batch: int, extra: list[str]) -> list[str]:
    return [PY, "-m", "ber.crossenc", "train", "--data-dir", args.data_dir, "--init", init,
            "--blocking", blocking_cache(), "--knn", "artifacts/emb/train_knn.npz",
            "--n-neg", "4", "--n-neg-knn", "3", "--max-pairs", str(pairs), "--batch", str(batch),
            "--model", out, *extra]  # fmt: skip


def step_ce(args, state) -> None:
    p = state["profile"]
    state["ce_batch"] = run_gpu(
        lambda b: ce_train_cmd(args, "models/e5-small-ft", "models/ce-small", p["ce_max_pairs"], b, []),
        "ce", p["ce_batch"], floor=16)  # fmt: skip


def step_evalpairs(args, state) -> None:
    run([PY, "scripts/make_ce_eval_pairs.py", "--blocking", blocking_cache(), "--data-dir",
         args.data_dir, "--entities", str(state["profile"]["eval_entities"])], "evalpairs")  # fmt: skip


def step_runB(args, state) -> None:
    a = state["run_A"]
    pipeline(args, state, "B", [f"scale.stage1_model={Path(a).as_posix()}/scale_model.joblib",
                                "scale.ce_model=models/ce-small"])  # fmt: skip


def step_pseudo(args, state) -> None:
    p = state["profile"]
    run([PY, "scripts/make_pseudo_labels.py", "--run", state["run_A"], "--run", state["run_B"],
         "--country", "France", "--data-dir", args.data_dir, "--max-pos", str(p["pseudo_pos"]),
         "--max-neg", str(p["pseudo_neg"]), "--out", "artifacts/pseudo_france.npz"], "pseudo")  # fmt: skip
    # unlabelled French pairs on which the gate measures how undecided a model is
    code = (
        "import numpy as np, pandas as pd, sys\n"
        "run, data = sys.argv[1], sys.argv[2]\n"
        "s = np.load(run + '/test_scores.npz')\n"
        "c = pd.read_csv(data + '/test/test_source1.tsv', sep='\\t', dtype=str, keep_default_na=False,"
        " quoting=3, encoding='utf-8-sig').set_index('entity_id')['country'].str.strip()"
        ".reindex(s['s1_ids']).to_numpy()\n"
        "fr = np.flatnonzero((c[s['rows']] == 'France') & (s['p'] > 0.02))\n"
        "pick = np.random.default_rng(0).choice(fr, min(100000, len(fr)), replace=False)\n"
        "np.savez('artifacts/france_ce_sample.npz', rows=s['rows'][pick], cols=s['cols'][pick])\n"
        "print(len(pick), 'French pairs')\n"
    )
    run([PY, "-c", code, state["run_B"], args.data_dir], "pseudo")


def gate(args, trained: str, source: str, final: str, tolerance: float, step: str) -> None:
    run([PY, "scripts/ce_gate.py", "--eval", "artifacts/ce_eval_pairs.npz", "--france",
         "artifacts/france_ce_sample.npz", "--data-dir", args.data_dir, "--tolerance",
         str(tolerance), "--report", f"runs/v6/{step}_gate.json",
         "--candidate", f"{trained}={source}={final}"], step)  # fmt: skip


def step_cefr(args, state) -> None:
    p = state["profile"]
    shutil.rmtree(ROOT / "models/ce-small-fr.tmp", ignore_errors=True)
    pseudo = ["--pseudo", "artifacts/pseudo_france.npz", "--pseudo-split", "test", "--lr", "3e-5"]
    run_gpu(lambda b: ce_train_cmd(args, "models/ce-small", "models/ce-small-fr.tmp",
                                   p["ce_fr_pairs"], b, pseudo),
            "cefr", state.get("ce_batch", p["ce_batch"]), floor=16)  # fmt: skip
    gate(args, "models/ce-small-fr.tmp", "models/ce-small", "models/ce-small-fr", 0.05, "cefr")


def step_large(args, state) -> None:
    p = state["profile"]
    if not args.large:
        return
    if not p["large_ok"]:
        say("large: skipped, under 14 GB free on the GPU")
        return
    if not (ROOT / "models/bge-reranker-v2-m3/config.json").exists():
        run([PY, "-m", "ber.cli", "fetch-model", "BAAI/bge-reranker-v2-m3", "--dest",
             "models/bge-reranker-v2-m3"], "large")  # fmt: skip
    shutil.rmtree(ROOT / "models/ce-bge-fr.tmp", ignore_errors=True)
    extra = ["--micro-batch", "64", "--freeze-embeddings", "--lr", "1.5e-5", "--pseudo",
             "artifacts/pseudo_france.npz", "--pseudo-split", "test"]  # fmt: skip
    run_gpu(lambda b: ce_train_cmd(args, "models/bge-reranker-v2-m3", "models/ce-bge-fr.tmp",
                                   1_000_000, b, extra), "large", 128, floor=32)  # fmt: skip
    gate(args, "models/ce-bge-fr.tmp", "models/ce-small", "models/ce-bge-fr", 0.10, "large")
    state["large"] = True


def step_groups(args, state) -> None:
    run([PY, "scripts/make_lw_groups.py", "--data-dir", args.data_dir, "--out",
         "artifacts/lw_groups_enc.npz", "--train", str(state["profile"]["lw_train"]),
         "--valid", "30000" if not args.smoke else "80"], "groups")  # fmt: skip


def step_pgroups(args, state) -> None:
    run([PY, "scripts/make_lw_groups.py", "--data-dir", args.data_dir, "--out",
         "artifacts/lwp_groups_enc.npz", "--side", "pool", "--train",
         str(state["profile"]["lwp_train"]), "--valid", "30000" if not args.smoke else "80"],
        "pgroups")  # fmt: skip


def listwise_train(args, state, groups: str, out: str, step: str) -> None:
    p = state["profile"]
    env = {"BER_LISTWISE_AMP": p["amp"]}

    def cmd(b):
        c = [PY, "-m", "ber.listwise", "train", "--data-dir", args.data_dir, "--groups", groups,
             "--init", "models/ce-small", "--model", out, "--batch", str(b),
             "--epochs", str(p["lw_epochs"]), "--lr", "3e-5"]  # fmt: skip
        return c + (["--grad-ckpt"] if p["lw_grad_ckpt"] else [])

    run_gpu(cmd, step, p["lw_batch"], env)


def step_lw(args, state) -> None:
    listwise_train(args, state, "artifacts/lw_groups_enc.npz", "models/lw", "lw")


def step_lwp(args, state) -> None:
    listwise_train(args, state, "artifacts/lwp_groups_enc.npz", "models/lwp", "lwp")


def step_lweval(args, state) -> None:
    n = str(state["profile"]["eval_entities"])
    for groups, model, name in (("artifacts/lw_groups_enc.npz", "models/lw", "lw"),
                                ("artifacts/lwp_groups_enc.npz", "models/lwp", "lwp")):  # fmt: skip
        run([PY, "-m", "ber.listwise", "eval", "--data-dir", args.data_dir, "--groups", groups,
             "--model", model, "--ce", "models/ce-small", "--max-entities", n], f"lweval_{name}")  # fmt: skip
        log = (WORK / "logs" / f"lweval_{name}.log").read_text(encoding="utf-8", errors="replace")
        report = log[log.rfind("\n{") + 1 :] if "\n{" in log else "{}"
        (WORK / f"lweval_{name}.json").write_text(report)


def scorer_models(state) -> str:
    models = ["models/ce-small-fr"]
    if state.get("large"):
        models.append("models/ce-bge-fr")
    return ",".join([*models, "models/lw", "models/lwp"])


def v6_sets(state) -> list[str]:
    sets = [f"scale.stage1_model={Path(state['run_A']).as_posix()}/scale_model.joblib",
            f"scale.ce_model={scorer_models(state)}", "scale.fingerprint=true"]  # fmt: skip
    if state.get("large"):
        sets.append("scale.ce_band=0.001,0.999")
    return sets


def step_runC(args, state) -> None:
    pipeline(args, state, "C", [*v6_sets(state), "scale.save_stage2=true"])


def search(args, state, only: str | None, out: str) -> None:
    p = state["profile"]
    cmd = [PY, "scripts/stage2_search.py", "--run", state["run_C"], "--threads",
           str(p["n_threads"]), "--max-sample", str(p["search_sample"]), "--out", out]  # fmt: skip
    cmd += ["--only", only] if only is not None else ["--mode", "greedy", "--monotone"]
    if args.smoke:
        cmd += ["--rounds", "100", "--stop", "10"]
    run(cmd, "search")


def step_search(args, state) -> None:
    run_c = Path(state["run_C"])
    names = json.loads((run_c / "stage2_inputs.json").read_text())
    fams = subprocess.run([PY, "-c", "import json,sys;from ber.scale import stage2_family as f;"
                           "print(','.join(sorted({f(n) for n in json.load(open(sys.argv[1]))})))",
                           str(run_c / "stage2_inputs.json")], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")}).stdout.strip().split(",")  # fmt: skip
    optional = [f for f in fams if f and f not in ("p1", "keep")]
    without = [f for f in optional if not f.startswith("lw")]
    search(args, state, ",".join(without), str(run_c / "search_without_listwise.tsv"))
    search(args, state, ",".join(optional), str(run_c / "search_everything.tsv"))
    search(args, state, None, str(run_c / "search_greedy.tsv"))
    state["stage2_names"] = len(names)


def read_tsv_rows(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def step_final(args, state) -> None:
    run_c = Path(state["run_C"])
    rows = []
    for name in ("search_greedy.tsv", "search_everything.tsv", "search_without_listwise.tsv"):
        if (run_c / name).exists():
            rows += read_tsv_rows(run_c / name)
    best = max(rows, key=lambda r: float(r["holdout"]))
    mono = [r for r in rows if r["monotone"] == "True" and r["families"] == best["families"]]
    use_mono = bool(mono) and float(mono[0]["holdout"]) >= float(best["holdout"]) - 0.0003
    exclude = best["exclude"]
    m = json.loads((run_c / "metrics.json").read_text())
    dec = m.get("decision", {})
    run_c_holdout = dec.get(dec.get("chosen", {}).get("strategy", ""), {}).get("holdout")
    if run_c_holdout is not None and float(best["holdout"]) < run_c_holdout - 0.01:
        # the offline search disagrees with run C itself by more than a point: do not trust
        # it; ship run C's own stage 2 (every family)
        say(f"final: search best {float(best['holdout']):.5f} is far below run C's "
            f"{run_c_holdout:.5f}; keeping every family")  # fmt: skip
        exclude, use_mono = "", False
    state["final_choice"] = {"families": best["families"], "exclude": exclude,
                             "holdout_search": float(best["holdout"]), "monotone": use_mono,
                             "run_c_holdout": run_c_holdout}  # fmt: skip
    say(f"final: {state['final_choice']}")
    pipeline(args, state, "final", [*v6_sets(state), f"scale.stage2_exclude={exclude}",
                                    f"scale.stage2_monotone={str(use_mono).lower()}"])  # fmt: skip
    run([PY, "-m", "ber.cli", "validate", "--matching", "output-final/matching_results.tsv",
         "--candidates", "output-final/candidate_pairs.tsv", "--test-dir",
         f"{args.data_dir}/test"], "final")  # fmt: skip


def metrics_line(run_dir: str | None) -> str:
    if not run_dir or not (Path(run_dir) / "metrics.json").exists():
        return "| (not run) |"
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    s1 = m.get("stage1", {}).get("holdout")
    s2 = m.get("stage2", {}).get("holdout")
    dec = m.get("decision", {})
    chosen = dec.get("chosen", {}).get("strategy")
    dh = dec.get(chosen, {}).get("holdout") if chosen else None
    test = m.get("test", {})
    fmt = lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
    return (f"| {Path(run_dir).name} | {fmt(s1)} | {fmt(s2)} | {fmt(dh)} ({chosen}) | "
            f"{test.get('validation_ok', '-')} | {test.get('predicted_singletons', '-')} |")  # fmt: skip


def step_report(args, state) -> None:
    lines = ["# v6 run: results", "",
             f"Machine: `{json.dumps(state.get('hardware', {}))}`", "",
             f"Profile: `{json.dumps(state.get('profile', {}))}`", "",
             "## Holdout macro F0.5 per run (orphan-simulated holdout, drop "
             f"{args.drop}; compare with v4 0.99001 and v5 in the handbook)", "",
             "| run | stage 1 | stage 2 | chosen decision | output valid | predicted singletons |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for key in ("run_A", "run_B", "run_C", "run_final"):
        lines.append(metrics_line(state.get(key)))
    run_c = Path(state["run_C"]) if state.get("run_C") else None
    if run_c:
        lines += ["", "## Stage-2 families (scripts/stage2_search.py, holdout macro F0.5)", ""]
        for name, title in (("search_without_listwise.tsv", "without C1/C7"),
                            ("search_everything.tsv", "with everything"),
                            ("search_greedy.tsv", "greedy forward selection")):  # fmt: skip
            if (run_c / name).exists():
                lines.append(f"**{title}**")
                lines.append("")
                for r in sorted(read_tsv_rows(run_c / name), key=lambda r: -float(r["holdout"]))[
                    :8
                ]:
                    lines.append(
                        f"- {float(r['holdout']):.5f}  {r['families']}  monotone={r['monotone']}"
                    )
                lines.append("")
    for name in ("lw", "lwp"):
        f = WORK / f"lweval_{name}.json"
        if f.exists():
            lines += [f"## {name} on held-out groups", "", "```", f.read_text()[:3000], "```", ""]
    for gate_file in ("cefr_gate.json", "large_gate.json"):
        f = WORK / gate_file
        if f.exists():
            lines += [f"## {gate_file}", "", "```", f.read_text()[:2000], "```", ""]
    lines += ["## What to submit", "",
              f"- `output-final/matching_results.tsv` (choice: `{json.dumps(state.get('final_choice', {}))}`)",
              "- fallbacks, weaker: `output-C/`, then `output-B/` (~v4), then `output-A/`",
              "- upload to the portal only after telling the lead; record the public score in the handbook"]  # fmt: skip
    (WORK / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    say(f"report: {WORK / 'RESULTS.md'}")


STEP_FUNCS = {name: globals()[f"step_{name}"] for name in STEPS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", required=True, help="the dataset folder holding train/ and test/")
    ap.add_argument("--only", default="", help="comma-separated steps to run (ignores markers)")
    ap.add_argument("--from", dest="start", default="", help="rerun from this step on")
    ap.add_argument("--large", action="store_true", help="also the France-adapted bge reranker")
    ap.add_argument("--drop", type=float, default=0.186, help="orphan simulation share")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, for synthetic data")
    args = ap.parse_args()
    args.data_dir = Path(args.data_dir).resolve().as_posix()
    for f in ("train/train_source1.tsv", "train/train_ground_truth.tsv", "test/test_source1.tsv"):
        if not Path(args.data_dir, f).exists():
            raise SystemExit(
                f"{args.data_dir}/{f} not found: --data-dir must hold train/ and test/"
            )
    WORK.mkdir(parents=True, exist_ok=True)
    state = load_state()
    # models/, artifacts/ and runs/v6/ are shared by every run in this checkout: a smoke
    # test's tiny embeddings or models must never be picked up by a real run, or the reverse
    kind = "smoke" if args.smoke else "real"
    if state.get("kind", kind) != kind or state.get("data_dir", args.data_dir) != args.data_dir:
        raise SystemExit(
            f"this checkout already holds a {state.get('kind')} run on {state.get('data_dir')}; "
            "use a fresh clone, or delete runs/v6, artifacts/ and models/ (keep "
            "models/multilingual-e5-small) before starting a different one"
        )
    state["kind"] = kind
    state["data_dir"] = args.data_dir
    if args.start:
        for s in STEPS[STEPS.index(args.start) :]:
            (WORK / "done" / s).unlink(missing_ok=True)
    todo = [s.strip() for s in args.only.split(",") if s.strip()] or STEPS
    for s in todo:
        if s not in STEP_FUNCS:
            raise SystemExit(f"unknown step {s!r}; steps: {', '.join(STEPS)}")
    t0 = time.time()
    for step in todo:
        if not args.only and done(step):
            say(f"{step}: done earlier, skipped")
            continue
        if step not in ("setup", "hardware") and "profile" not in state:
            step_hardware(args, state)
        say(f"===== {step}")
        STEP_FUNCS[step](args, state)
        mark(step)
        save_state(state)
    say(f"all steps done in {(time.time() - t0) / 3600:.1f} h; see {WORK / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
