#!/usr/bin/env python3
"""Build a relation-only probe using SAMJAM's tracked object IDs as constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT = """<image>
You are the relation-generation backend of a video scene graph system.

The current frame contains the following already detected and tracked objects:
obj[{count}]{{id,name}}:
{object_rows}

Generate relationships ONLY between these known objects.

Output Format:
<answer>
rel[M]{{subj,pred,obj}}:
  subj,pred,obj
  ...
</answer>

Rules:
- Use the exact integer IDs listed above. Do not invent, renumber, or omit IDs inside a relation.
- Do not output an object table or natural-language explanation.
- Output only relationships clearly supported by the image.
- Prefer predicates such as on, under, beside, near, in-front-of, behind, inside, holding, touching.
- If no relationship is clearly supported, output rel[0]{{subj,pred,obj}}: with no rows.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--samjam-graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(args.images_dir.glob("*.jpg"))
    records = []
    for index, image in enumerate(images):
        object_path = args.samjam_graph_dir / f"{index}_objs.json"
        if not object_path.exists():
            continue
        objects = json.loads(object_path.read_text(encoding="utf-8"))
        object_rows = "\n".join(f"  {item['id']},{item['name']}" for item in objects)
        prompt = PROMPT.format(count=len(objects), object_rows=object_rows)
        records.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<answer>\nrel[0]{subj,pred,obj}:\n</answer>"},
                ],
                "images": [str(image.resolve())],
                "known_object_ids": [int(item["id"]) for item in objects],
                "known_objects": objects,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Prepared {len(records)} constrained frames at {args.output}")


if __name__ == "__main__":
    main()
