"""
ANPR OCR inference from YOLO plate-coordinate JSON.

Input JSON format:
[
  {
    "image": "/root/video_recorder/snapshots/vehicles/NumberPlate.JPG",
    "width": 1142,
    "height": 1189,
    "detections": [
      {"x1": 551.31, "y1": 541.55, "x2": 706.43, "y2": 592.64, "confidence": 0.78}
    ]
  }
]

Usage:
  python anpr_test.py --detections_json results/plate_coordinates.json \
      --device cpu --output_json results/anpr_results.json

This script reads the original vehicle images, crops only the plate regions
listed in the JSON, and runs the existing PaddleOCR recognition model on each
crop. It does not run YOLO or use an RTSP stream.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import re
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent

CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-05},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    },
}

IMAGE_SHAPE = [3, 48, 320]


def _find_paddleocr(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates += [
        _SCRIPT_DIR / "PaddleOCR",
        _SCRIPT_DIR,
        Path.cwd(),
        Path.cwd() / "PaddleOCR",
    ]
    for candidate in candidates:
        if (candidate / "ppocr" / "__init__.py").is_file():
            return candidate
    return None


def _ensure_paddleocr(explicit_path=None):
    root = _find_paddleocr(explicit_path)
    if root is None:
        clone_target = _SCRIPT_DIR / "PaddleOCR"
        print(f"ppocr not found. Cloning PaddleOCR into {clone_target} ...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/PaddlePaddle/PaddleOCR.git",
            str(clone_target),
        ])
        root = clone_target
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on plate crops specified by YOLO detection JSON."
    )
    parser.add_argument(
        "--detections_json", required=True,
        help="YOLO output JSON, e.g. results/plate_coordinates.json",
    )
    parser.add_argument("--weights", default="", help="Path to model.safetensors.")
    parser.add_argument("--dict_path", default="", help="Path to en_dict.txt.")
    parser.add_argument(
        "--device", default="gpu", choices=["gpu", "cpu"],
        help="Inference device; falls back to CPU if Paddle has no CUDA support.",
    )
    parser.add_argument(
        "--output_json", default="results/anpr_results.json",
        help="Output JSON path (default: results/anpr_results.json).",
    )
    parser.add_argument("--paddleocr_dir", default="", help="PaddleOCR repo root.")
    parser.add_argument(
        "--padding", type=int, default=2,
        help="Extra pixels around each detected plate crop (default: 2).",
    )
    return parser.parse_args()


def resolve_path(user_path: str, filename: str) -> str:
    if user_path and os.path.exists(user_path):
        return user_path
    fallback = _SCRIPT_DIR / filename
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(
        f"Could not find {filename}. Put it next to this script or pass its path."
    )


def load_safetensors_to_paddle(paddle_mod, weight_path: str):
    from safetensors.numpy import load_file
    np_state = load_file(weight_path)
    return {key: paddle_mod.to_tensor(value) for key, value in np_state.items()}


def resize_for_rec(img_bgr, target_shape):
    _, height, width = target_shape
    img_h, img_w = img_bgr.shape[:2]
    ratio = height / img_h
    new_w = min(int(img_w * ratio), width)
    resized = cv2.resize(img_bgr, (new_w, height))
    if new_w < width:
        padded = np.zeros((height, width, 3), dtype=np.uint8)
        padded[:, :new_w, :] = resized
        resized = padded
    return resized


def preprocess(img_bgr, target_shape):
    img = resize_for_rec(img_bgr, target_shape)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose((2, 0, 1))


def load_detections_json(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Detection JSON must contain a list of image records.")
    return data


def run_ocr(model, post_process, paddle, crop_bgr):
    tensor = paddle.to_tensor(
        np.expand_dims(preprocess(crop_bgr, IMAGE_SHAPE), axis=0)
    )
    with paddle.no_grad():
        predictions = model(tensor)

    if isinstance(predictions, dict):
        pred_tensor = predictions.get("ctc", next(iter(predictions.values())))
    elif isinstance(predictions, (list, tuple)):
        pred_tensor = predictions[0]
    else:
        pred_tensor = predictions

    post_result = post_process(pred_tensor.numpy())
    if isinstance(post_result, (list, tuple)) and post_result:
        text, confidence = post_result[0]
    else:
        text, confidence = "", 0.0
    return text.strip().upper(), float(confidence)


def main():
    args = parse_args()
    _ensure_paddleocr(args.paddleocr_dir or None)

    import paddle
    from ppocr.modeling.architectures import build_model as ppocr_build_model
    from ppocr.postprocess import build_post_process

    if args.device == "gpu" and not paddle.is_compiled_with_cuda():
        print("CUDA not available, falling back to CPU.")
        paddle.set_device("cpu")
    else:
        paddle.set_device(args.device)

    weights_path = resolve_path(args.weights, "model.safetensors")
    dict_path = resolve_path(args.dict_path, "en_dict.txt")

    post_process = build_post_process({
        "name": "CTCLabelDecode",
        "character_dict_path": dict_path,
        "use_space_char": True,
    })

    config = copy.deepcopy(MODEL_CONFIG)
    model = ppocr_build_model(config["Architecture"])
    model.eval()
    state_dict = load_safetensors_to_paddle(paddle, weights_path)
    model.set_state_dict(state_dict)
    print(f"Loaded weights from {weights_path}")

    records = load_detections_json(args.detections_json)
    print(f"Loaded detection JSON: {args.detections_json}")
    print(f"Image records: {len(records)}")

    results = []
    total_detections = 0
    for record in records:
        image_value = record.get("image", "")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = (_SCRIPT_DIR / image_path).resolve()

        detections = record.get("detections", [])
        if not detections:
            print(f"[SKIP] {image_path.name}: no plate detections")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(
                f"[WARNING] Cannot read image: {image_path}. "
                "Check file permissions and path."
            )
            for det_index, det in enumerate(detections):
                results.append({
                    "image": image_path.name,
                    "plate_index": det_index,
                    "plate_coordinates": det,
                    "detector_confidence": det.get("confidence"),
                    "prediction": "",
                    "ocr_confidence": 0.0,
                    "error": "image_unreadable_or_permission_denied",
                })
            continue

        img_h, img_w = image.shape[:2]
        for det_index, det in enumerate(detections):
            try:
                x1 = int(np.floor(float(det["x1"]))) - args.padding
                y1 = int(np.floor(float(det["y1"]))) - args.padding
                x2 = int(np.ceil(float(det["x2"]))) + args.padding
                y2 = int(np.ceil(float(det["y2"]))) + args.padding
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[WARNING] Invalid detection in {image_path.name}: {exc}")
                continue

            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            if x2 <= x1 or y2 <= y1:
                print(f"[WARNING] Invalid/empty crop: {image_path.name} plate {det_index}")
                continue

            plate_crop = image[y1:y2, x1:x2].copy()
            if plate_crop.size == 0:
                continue

            try:
                text, ocr_confidence = run_ocr(model, post_process, paddle, plate_crop)

                # Rename only when this image has one detected plate.
                if len(detections) == 1 and text and ocr_confidence >= 0.90:
                    plate_number = re.sub(r"[^A-Z0-9]", "", text.upper())
                    if plate_number:
                        new_path = image_path.with_name(
                            plate_number + image_path.suffix.lower()
                        )
                        if new_path.resolve() == image_path.resolve():
                            print(f"[INFO] Already named: {image_path.name}")
                        elif new_path.exists():
                            print(f"[WARNING] Rename skipped; file exists: {new_path.name}")
                        else:
                            try:
                                image_path.rename(new_path)
                                print(f"[RENAMED] {image_path.name} -> {new_path.name}")
                                image_path = new_path
                            except OSError as rename_error:
                                print(f"[WARNING] Rename failed: {rename_error}")

                item = {
                    "image": image_path.name,
                    "plate_index": det_index,
                    "plate_coordinates": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    },
                    "detector_confidence": round(float(det.get("confidence", 0.0)), 4),
                    "prediction": text,
                    "ocr_confidence": round(ocr_confidence, 4),
                }
                results.append(item)
                total_detections += 1
                print(
                    f"[OCR] {image_path.name} plate {det_index + 1}: "
                    f"{text or '<no text>'} "
                    f"(det={item['detector_confidence']:.4f}, "
                    f"ocr={item['ocr_confidence']:.4f})"
                )
            except Exception as exc:
                print(f"[ERROR] OCR failed for {image_path.name} plate {det_index + 1}: {exc}")
                results.append({
                    "image": image_path.name,
                    "plate_index": det_index,
                    "plate_coordinates": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    },
                    "detector_confidence": round(float(det.get("confidence", 0.0)), 4),
                    "prediction": "",
                    "ocr_confidence": 0.0,
                    "error": str(exc),
                })

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = (_SCRIPT_DIR / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n[INFO] OCR processed plate crops: {total_detections}")
    print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
