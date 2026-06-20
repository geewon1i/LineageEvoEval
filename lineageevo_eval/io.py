"""Input and output helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class InputFormatError(ValueError):
    """Raised when the JSONL input cannot be parsed."""


@dataclass(frozen=True)
class FactorRecord:
    input_index: int
    factor_id: str
    expression: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_factor_records(path: str | Path) -> list[FactorRecord]:
    input_path = Path(path)
    if not input_path.exists():
        raise InputFormatError(f"input file not found: {input_path}")

    records: list[FactorRecord] = []
    with input_path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputFormatError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise InputFormatError(f"line {line_no}: each JSONL line must be an object")
            expression = str(item.get("expression", "")).strip()
            if not expression:
                raise InputFormatError(f"line {line_no}: missing required field 'expression'")
            factor_id = str(item.get("factor_id") or f"factor_{len(records) + 1:03d}")
            metadata = {key: value for key, value in item.items() if key not in {"factor_id", "expression"}}
            records.append(
                FactorRecord(
                    input_index=len(records) + 1,
                    factor_id=factor_id,
                    expression=expression,
                    metadata=metadata,
                )
            )
    if not records:
        raise InputFormatError(f"input file is empty: {input_path}")
    return records


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "input_index",
        "factor_id",
        "selection_rank",
        "baseline",
        "author",
        "expression",
        "qlib_expression",
        "status",
        "failure_reason",
        "orientation",
        "test_ic",
        "test_icir",
    ]
    seen = set()
    fields: list[str] = []
    for name in preferred:
        if any(name in row for row in rows):
            fields.append(name)
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fields.append(name)
                seen.add(name)
    return fields
