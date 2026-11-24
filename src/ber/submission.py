"""Output files and a local validator mirroring the published rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ber import schema
from ber.io import read_tsv, write_id_lists


def write_outputs(
    out_dir: str | Path,
    matches: Mapping[str, Sequence[str]],
    candidates: Mapping[str, Sequence[str]],
    s1_order: Sequence[str],
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    m = write_id_lists(out_dir / schema.MATCHING_FILE, matches, s1_order, schema.MATCHING_HEADER)
    c = write_id_lists(
        out_dir / schema.CANDIDATE_FILE, candidates, s1_order, schema.CANDIDATE_HEADER
    )
    return m, c


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = ["PASS" if self.ok else f"FAIL ({len(self.errors)} errors)"]
        lines += [f"  {i}. ERROR {e}" for i, e in enumerate(self.errors, 1)]
        lines += [f"  - WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def _check_file(
    path: Path,
    header: tuple[str, str],
    s1_ids: set[str],
    pool_ids: set[str],
    report: ValidationReport,
) -> dict[str, list[str]]:
    label = path.name
    if not path.exists():
        report.errors.append(f"{label}: file not found at {path}")
        return {}
    raw = path.read_bytes()
    if b"\r\n" in raw:
        report.warnings.append(f"{label}: CRLF line endings (LF expected)")
    if b'"' in raw:
        report.errors.append(f"{label}: contains quote characters; id lists must be unquoted")
    text = raw.decode("utf-8-sig")
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln != ""]
    if not lines:
        report.errors.append(f"{label}: empty file")
        return {}
    got_header = tuple(lines[0].split("\t"))
    if got_header != header:
        report.errors.append(f"{label}: header {got_header} != expected {header}")
    rows: dict[str, list[str]] = {}
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) == 1:
            parts.append("")
        if len(parts) != 2:
            report.errors.append(f"{label}:{n}: expected 2 tab-separated columns, got {len(parts)}")
            continue
        key, value = parts[0].strip(), parts[1].strip()
        if key in rows:
            report.errors.append(f"{label}:{n}: duplicate source1_entity_id {key}")
            continue
        if key not in s1_ids:
            report.errors.append(f"{label}:{n}: {key} is not a Source 1 id of the test set")
        ids = [t.strip() for t in value.split(",")] if value else []
        if any(not t for t in ids):
            report.errors.append(f"{label}:{n}: empty element in id list {value!r}")
        ids = [t for t in ids if t]
        if len(ids) != len(set(ids)):
            report.errors.append(f"{label}:{n}: duplicate ids within the list for {key}")
        for t in ids:
            if t.startswith(schema.SOURCE_PREFIX[1]):
                report.errors.append(f"{label}:{n}: {key} lists Source 1 id {t}")
            elif t not in pool_ids:
                report.errors.append(f"{label}:{n}: {t} is not a Source 2/3 id of the test set")
        rows[key] = ids
    missing = s1_ids.difference(rows)
    if missing:
        sample = sorted(missing)[:5]
        report.errors.append(f"{label}: {len(missing)} Source 1 entities missing, e.g. {sample}")
    return rows


def validate_outputs(
    matching: str | Path, candidates: str | Path, test_dir: str | Path
) -> ValidationReport:
    """Check both files against every rule in the problem statement."""
    test_dir = Path(test_dir)
    split = test_dir.name if test_dir.name in {"train", "test"} else "test"
    s1 = read_tsv(test_dir / schema.source_file(split, 1))[schema.ENTITY_ID]
    s2 = read_tsv(test_dir / schema.source_file(split, 2))[schema.ENTITY_ID]
    s3 = read_tsv(test_dir / schema.source_file(split, 3))[schema.ENTITY_ID]
    s1_ids, pool_ids = set(s1), set(s2) | set(s3)
    report = ValidationReport()
    m_rows = _check_file(Path(matching), schema.MATCHING_HEADER, s1_ids, pool_ids, report)
    c_rows = _check_file(Path(candidates), schema.CANDIDATE_HEADER, s1_ids, pool_ids, report)
    if m_rows and c_rows:
        outside = [
            (k, t) for k, ids in m_rows.items() for t in ids if t not in set(c_rows.get(k, ()))
        ]
        if outside:
            report.warnings.append(
                f"{len(outside)} matched ids never appeared as candidates (pipeline bug), e.g. {outside[:3]}"
            )
    return report
