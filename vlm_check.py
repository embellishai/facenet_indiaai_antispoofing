#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import ollama

FACE_CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "front_facing": {"type": "boolean"},
        "front_facing_score": {"type": "number"},
        "both_eyes_visible": {"type": "boolean"},
        "both_ears_visible": {"type": "boolean"},
        "nose_visible": {"type": "boolean"},
        "chin_visible": {"type": "boolean"},
        "head_top_visible": {"type": "boolean"},
        "notes": {"type": "string"}
    },
    "required": [
        "front_facing","front_facing_score","both_eyes_visible","both_ears_visible",
        "nose_visible","chin_visible","head_top_visible"
    ]
}

SYSTEM_GUIDELINES = (
    "You are a strict photo quality inspector for ID-style face captures. "
    "Evaluate if the face is front-facing and whether all required features are visible: head top, chin, nose, both eyes, both ears. "
    "Output only JSON that conforms to the provided schema. "
)

USER_TASK = (
    "Return an object with fields: front_facing, front_facing_score (0..1), both_eyes_visible, both_ears_visible, "
    "nose_visible, chin_visible, head_top_visible, notes."
)


def check_image_with_qwen_ollama(image_path: Path, model: str) -> dict:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_GUIDELINES},
            {"role": "user", "content": USER_TASK, "images": [str(image_path)]}
        ],
        format=FACE_CHECK_SCHEMA,
        options={"temperature": 0.0}
    )
    content = resp["message"]["content"]
    return json.loads(content)


def decide_pass(result: dict, min_front_score: float = 0.8) -> bool:
    req = [
        bool(result.get("front_facing", False)),
        bool(result.get("both_eyes_visible", False)),
        bool(result.get("both_ears_visible", False)),
        bool(result.get("nose_visible", False)),
        bool(result.get("chin_visible", False)),
        bool(result.get("head_top_visible", False)),
    ]
    return all(req) and float(result.get("front_facing_score", 0.0)) >= min_front_score


def main():
    parser = argparse.ArgumentParser(description='VLM guideline checker via Ollama (Qwen2.5-VL)')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', default='qwen2.5vl:3b')
    parser.add_argument('--min-front', type=float, default=0.8)
    args = parser.parse_args()

    img_path = Path(args.image)
    res = check_image_with_qwen_ollama(img_path, args.model)
    passed = decide_pass(res, args.min_front)
    print(json.dumps({"image": str(img_path), "passed": passed, "result": res}, indent=2))


if __name__ == '__main__':
    main()
