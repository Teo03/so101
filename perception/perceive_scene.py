#!/usr/bin/env python3
"""Prompted, zero-shot scene perception for the SO-101 tabletop.

This module does not open a robot serial port.  It uses YOLOE instance
segmentation to find task objects from text prompts, then records pixel-space
geometry that a calibrated workspace transform can convert to robot XY.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOCAL_VISION_PACKAGES = ROOT / ".vision_packages"
if LOCAL_VISION_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_VISION_PACKAGES))

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
DEFAULT_MODEL = ROOT / "models/yoloe-26n-seg.pt"
DEFAULT_IMAGE = ROOT / "runtime/outputs/scene_live.png"
DEFAULT_JSON = ROOT / "runtime/outputs/scene_observation.json"
DEFAULT_ANNOTATED = Path.home() / "Desktop/so101_scene_perception.png"

# Synonyms are deliberately grouped into stable skill-level names.  A new task
# can change these prompts without changing camera geometry, IK, or execution.
PROMPT_GROUPS = {
    "bar": ("wrapped chocolate bar", "chocolate bar", "candy bar"),
    "basket": ("wooden basket", "wicker basket", "storage basket"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--image", type=Path, help="Use an existing 640x480 image.")
    parser.add_argument("--capture", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--annotated", type=Path, default=DEFAULT_ANNOTATED)
    return parser.parse_args()


def capture_frame(camera: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-video_size",
            f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
            "-framerate",
            "30",
            "-i",
            camera,
            "-frames:v",
            "1",
            "-y",
            str(output),
        ],
        check=True,
    )


def normalize_axis_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def mask_geometry(mask: np.ndarray) -> dict[str, object]:
    binary = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("YOLOE returned an empty segmentation mask")
    contour = max(contours, key=cv2.contourArea)
    center, size, angle = cv2.minAreaRect(contour)
    long_side, short_side = size
    if long_side < short_side:
        long_side, short_side = short_side, long_side
        angle += 90.0
    angle = normalize_axis_angle_deg(angle)
    box = cv2.boxPoints((center, (long_side, short_side), angle))
    return {
        "center_px": [float(center[0]), float(center[1])],
        "size_px": [float(long_side), float(short_side)],
        "long_axis_angle_deg": angle,
        "area_px": float(cv2.contourArea(contour)),
        "box_px": box.astype(float).tolist(),
        "contour_px": contour[:, 0, :].astype(int).tolist(),
    }


def semantic_name(prompt: str) -> str:
    for name, prompts in PROMPT_GROUPS.items():
        if prompt in prompts:
            return name
    raise KeyError(prompt)


def candidate_score(candidate: dict[str, object]) -> float:
    long_side, short_side = candidate["size_px"]
    aspect = long_side / max(short_side, 1.0)
    confidence = float(candidate["confidence"])
    area = float(candidate["area_px"])
    if candidate["semantic_name"] == "bar":
        if aspect < 1.8 or not 800.0 <= area <= 24000.0:
            return -1.0
        return confidence * min(aspect / 2.5, 1.6)
    if area < 3500.0 or aspect > 3.0:
        return -1.0
    return confidence * min(np.sqrt(area / 6000.0), 2.0)


def run_perception(
    image_path: Path,
    model_path: Path,
    confidence: float,
    device: str,
) -> dict[str, object]:
    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise RuntimeError(
            "YOLOE is unavailable. Install Ultralytics into .vision_packages "
            "or run with PYTHONPATH pointing at an existing installation."
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(f"Expected a readable {IMAGE_WIDTH}x{IMAGE_HEIGHT} image: {image_path}")

    prompts = [prompt for group in PROMPT_GROUPS.values() for prompt in group]
    model = YOLOE(str(model_path))
    # Ultralytics resolves the MobileCLIP text encoder relative to the current
    # directory. Keep that large ignored asset in the repository root.
    with contextlib.chdir(ROOT):
        model.set_classes(prompts)
    result = model.predict(
        image,
        conf=confidence,
        imgsz=640,
        device=device,
        retina_masks=True,
        verbose=False,
    )[0]

    candidates: list[dict[str, object]] = []
    if result.boxes is not None and result.masks is not None:
        masks = result.masks.data.detach().cpu().numpy()
        for box, mask in zip(result.boxes, masks, strict=True):
            class_id = int(box.cls.item())
            prompt = prompts[class_id]
            geometry = mask_geometry(mask)
            candidate = {
                "semantic_name": semantic_name(prompt),
                "prompt": prompt,
                "confidence": float(box.conf.item()),
                "bbox_xyxy_px": [float(value) for value in box.xyxy[0].tolist()],
                **geometry,
            }
            candidate["selection_score"] = candidate_score(candidate)
            candidates.append(candidate)

    selected: dict[str, dict[str, object]] = {}
    for semantic in PROMPT_GROUPS:
        valid = [
            candidate
            for candidate in candidates
            if candidate["semantic_name"] == semantic and candidate["selection_score"] >= 0.0
        ]
        if not valid:
            raise RuntimeError(f"No geometrically valid YOLOE detection for {semantic!r}")
        selected[semantic] = max(valid, key=lambda candidate: candidate["selection_score"])

    return {
        "format": "so101-scene-observation-v1",
        "image": str(image_path),
        "model": str(model_path),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "prompts": prompts,
        "selected": selected,
        "candidates": candidates,
    }


def annotate(image_path: Path, observation: dict[str, object], output: Path) -> None:
    image = cv2.imread(str(image_path))
    selected_ids = {id(value) for value in observation["selected"].values()}
    for candidate in observation["candidates"]:
        contour = np.asarray(candidate["contour_px"], dtype=np.int32)
        is_selected = id(candidate) in selected_ids
        color = (0, 210, 0) if is_selected else (120, 120, 120)
        cv2.polylines(image, [contour], True, color, 2 if is_selected else 1)
        center = tuple(np.rint(candidate["center_px"]).astype(int))
        label = (
            f"{candidate['semantic_name']} {candidate['confidence']:.2f}"
            if is_selected
            else f"reject {candidate['prompt']} {candidate['confidence']:.2f}"
        )
        cv2.putText(
            image,
            label,
            (max(4, center[0] - 55), max(18, center[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            image,
            label,
            (max(4, center[0] - 55), max(18, center[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        if is_selected:
            cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Could not write {output}")


def main() -> None:
    args = parse_args()
    image_path = args.image
    if image_path is None:
        capture_frame(args.camera, args.capture)
        image_path = args.capture
    observation = run_perception(image_path, args.model, args.confidence, args.device)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(observation, indent=2) + "\n")
    annotate(image_path, observation, args.annotated)
    summary = {
        name: {
            "prompt": value["prompt"],
            "confidence": round(value["confidence"], 3),
            "center_px": [round(number, 1) for number in value["center_px"]],
            "angle_deg": round(value["long_axis_angle_deg"], 1),
        }
        for name, value in observation["selected"].items()
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[vision] observation={args.json_output} annotated={args.annotated}", flush=True)


if __name__ == "__main__":
    main()
