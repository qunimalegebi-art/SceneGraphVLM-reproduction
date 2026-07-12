#!/usr/bin/env python3
"""Measure whether SceneGraphVLM obeys a supplied object-ID whitelist."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REL_HEADER = re.compile(r"rel\[\d+\]\{subj,pred,obj\}:?", re.IGNORECASE)


def parse_relation_rows(text: str) -> tuple[list[tuple[int, str, int]], int]:
    match = REL_HEADER.search(text)
    if not match:
        return [], 0
    relations = []
    malformed = 0
    for raw_line in text[match.end() :].splitlines():
        line = raw_line.strip().strip("`")
        if not line or line.startswith("</answer>"):
            continue
        fields = [part.strip() for part in line.lstrip("- ").split(",")]
        if len(fields) != 3:
            malformed += 1
            continue
        try:
            relations.append((int(fields[0]), fields[1], int(fields[2])))
        except ValueError:
            malformed += 1
    return relations, malformed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line]
    predictions = [json.loads(line) for line in args.pred_jsonl.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    valid_relations = 0
    total_relations = 0
    malformed_relations = 0
    for index, (source, prediction) in enumerate(zip(inputs, predictions)):
        known_ids = set(source.get("known_object_ids", []))
        text = prediction.get("predict", "")
        relations, malformed = parse_relation_rows(text)
        valid = sum(subject in known_ids and obj in known_ids for subject, _, obj in relations)
        valid_relations += valid
        total_relations += len(relations)
        malformed_relations += malformed
        rows.append(
            {
                "frame": index,
                "relation_only": "obj[" not in text.lower(),
                "relations": len(relations),
                "valid_whitelist_relations": valid,
                "malformed_relation_rows": malformed,
                "generation_time_sec": prediction.get("gen_time_sec"),
            }
        )

    summary = {
        "frames": len(rows),
        "relation_only_compliance_rate": (
            sum(row["relation_only"] for row in rows) / len(rows) if rows else 0.0
        ),
        "valid_whitelist_relation_rate": (
            valid_relations / total_relations if total_relations else 0.0
        ),
        "malformed_relation_rows": malformed_relations,
        "mean_generation_time_sec": (
            sum(float(row["generation_time_sec"] or 0) for row in rows) / len(rows) if rows else 0.0
        ),
        "frames_detail": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
