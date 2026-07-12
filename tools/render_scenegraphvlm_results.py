#!/usr/bin/env python3
"""Parse SceneGraphVLM TOON predictions and create report-friendly visuals."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OBJ_HEADER = re.compile(r"obj\[\d+\]\{id,name,x1,y1,x2,y2\}:?", re.IGNORECASE)
REL_HEADER = re.compile(r"rel\[\d+\]\{subj,pred,obj\}:?", re.IGNORECASE)
PALETTE = [
    "#00C2FF", "#FFB000", "#7CE577", "#FF6B9A", "#A78BFA",
    "#FF7043", "#26C6DA", "#D4E157", "#EC407A", "#42A5F5",
]


def answer_text(text: str) -> str:
    if "<answer>" in text:
        text = text.split("<answer>", 1)[1]
    if "</answer>" in text:
        text = text.split("</answer>", 1)[0]
    return text.strip()


def parse_toon(text: str) -> tuple[list[dict], list[dict]]:
    text = answer_text(text)
    object_match = OBJ_HEADER.search(text)
    relation_match = REL_HEADER.search(text)
    if not object_match:
        return [], []
    object_block = text[object_match.end() : relation_match.start() if relation_match else len(text)]
    relation_block = text[relation_match.end() :] if relation_match else ""

    objects = []
    for line in object_block.splitlines():
        fields = [part.strip() for part in line.strip().lstrip("- ").split(",")]
        if len(fields) < 6:
            continue
        try:
            object_id = int(fields[0])
            x1, y1, x2, y2 = map(lambda value: int(float(value)), fields[-4:])
        except ValueError:
            continue
        objects.append(
            {"id": object_id, "name": ",".join(fields[1:-4]).strip(), "box": [x1, y1, x2, y2]}
        )

    relations = []
    for line in relation_block.splitlines():
        fields = [part.strip() for part in line.strip().lstrip("- ").split(")")]
        # Normal TOON rows are comma separated; keep this fallback isolated from object names.
        if len(fields) == 1:
            fields = [part.strip() for part in line.strip().lstrip("- ").split(",")]
        if len(fields) < 3:
            continue
        try:
            subject = int(fields[0])
            obj = int(fields[-1])
        except ValueError:
            continue
        relations.append({"subject": subject, "predicate": ",".join(fields[1:-1]).strip(), "object": obj})
    return objects, relations


def text_font(size: int = 16):
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_frame(image_path: Path, objects: list[dict], relations: list[dict], output_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    panel_width = 390
    canvas = Image.new("RGB", (image.width + panel_width, image.height), "#111827")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = text_font(16)
    small = text_font(14)
    names = {item["id"]: item["name"] for item in objects}

    for item in objects:
        color = PALETTE[item["id"] % len(PALETTE)]
        x1, y1, x2, y2 = item["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width - 1, x2), min(image.height - 1, y2)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{item['id']}: {item['name']}"
        box = draw.textbbox((x1, y1), label, font=small)
        label_height = box[3] - box[1] + 5
        draw.rectangle((x1, max(0, y1 - label_height), x1 + box[2] - box[0] + 6, y1), fill=color)
        draw.text((x1 + 3, max(0, y1 - label_height + 1)), label, fill="black", font=small)

    offset = image.width + 16
    draw.text((offset, 14), "SceneGraphVLM", fill="#F9FAFB", font=text_font(21))
    draw.text((offset, 50), f"Objects: {len(objects)}", fill="#93C5FD", font=font)
    draw.text((offset + 150, 50), f"Relations: {len(relations)}", fill="#86EFAC", font=font)
    y = 84
    for relation in relations[:19]:
        subject_name = names.get(relation["subject"], "UNKNOWN")
        object_name = names.get(relation["object"], "UNKNOWN")
        line = f"{relation['subject']}:{subject_name}"
        line2 = f"  --{relation['predicate']}--> {relation['object']}:{object_name}"
        draw.text((offset, y), line, fill="#E5E7EB", font=small)
        draw.text((offset, y + 17), line2, fill="#FBBF24", font=small)
        y += 39
        if y > image.height - 35:
            break
    if len(relations) > 19:
        draw.text((offset, image.height - 25), f"... +{len(relations) - 19} more", fill="#9CA3AF", font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.pred_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    triples = []
    rendered = []
    predicate_counts: Counter[str] = Counter()
    valid_relations = 0
    all_relations = 0
    parsed_frames = 0
    object_counts = []
    relation_counts = []
    generation_times = []

    for index, record in enumerate(records):
        prediction = record.get("predict", "")
        objects, relations = parse_toon(prediction)
        if objects:
            parsed_frames += 1
        object_counts.append(len(objects))
        relation_counts.append(len(relations))
        if record.get("gen_time_sec") is not None:
            generation_times.append(float(record["gen_time_sec"]))
        ids = {item["id"] for item in objects}
        for relation in relations:
            all_relations += 1
            is_valid = relation["subject"] in ids and relation["object"] in ids
            valid_relations += int(is_valid)
            predicate_counts[relation["predicate"]] += 1
            triples.append({"frame": index, **relation, "valid_ids": is_valid})
        images = record.get("images") or []
        if images and Path(images[0]).is_file():
            rendered.append(
                render_frame(
                    Path(images[0]), objects, relations, args.output_dir / "frames" / f"frame_{index:04d}.jpg"
                )
            )

    if rendered:
        rendered[0].save(
            args.output_dir / "scenegraphvlm.gif",
            save_all=True,
            append_images=rendered[1:],
            duration=900,
            loop=0,
            optimize=False,
        )
    with (args.output_dir / "triples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "subject", "predicate", "object", "valid_ids"])
        writer.writeheader()
        writer.writerows(triples)
    summary = {
        "frames": len(records),
        "parsed_frames": parsed_frames,
        "parse_success_rate": parsed_frames / len(records) if records else 0.0,
        "mean_objects_per_frame": sum(object_counts) / len(object_counts) if object_counts else 0.0,
        "mean_relations_per_frame": sum(relation_counts) / len(relation_counts) if relation_counts else 0.0,
        "mean_generation_time_sec": (
            sum(generation_times) / len(generation_times) if generation_times else None
        ),
        "relations": all_relations,
        "valid_relation_id_rate": valid_relations / all_relations if all_relations else 0.0,
        "predicate_counts": dict(predicate_counts.most_common()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
