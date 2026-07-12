#!/usr/bin/env python3
"""Prepare a small frame folder as Swift JSONL for SceneGraphVLM probing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


PROMPT = """<image>
Generate a structured scene graph for an image of size ({width} x {height}) using the following text format.

Output Format:

<answer>
obj[N]{{id,name,x1,y1,x2,y2}}:
  id,name,x1,y1,x2,y2
  ...
rel[M]{{subj,pred,obj}}:
  subj,pred,obj
  ...
</answer>

Guidelines:
- Objects:
  - Use integer IDs starting from 1 in the id field.
  - The name must be the object category name.
  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.
  - Include all visible objects, even if they have no relationships.
- Relationships:
  - Represent interactions using integer object IDs in subj and obj.
  - pred is the relationship type, such as in-front-of, attached-to, beside.
  - Omit relationships for objects that do not participate in any interaction.

Now generate the complete scene graph for the provided image. Write your response only between <answer> and </answer> tags."""


def image_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    frames = sorted(
        [path for path in args.frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=image_sort_key,
    )
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"No image frames found in {args.frames_dir}")

    image_dir = args.output_dir / "frames_640x480"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = []
    for index, source in enumerate(frames):
        target = image_dir / f"{index:04d}.jpg"
        with Image.open(source) as image:
            image.convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS).save(
                target, quality=95
            )
        prompt = PROMPT.format(width=args.width, height=args.height)
        records.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<answer>\n</answer>"},
                ],
                "images": [str(target.resolve())],
            }
        )
        manifest.append({"frame": index, "source": str(source.resolve()), "prepared": str(target.resolve())})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "probe.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(records)} frames at {args.output_dir}")


if __name__ == "__main__":
    main()
