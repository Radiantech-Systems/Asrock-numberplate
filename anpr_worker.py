#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/asri/yolo11-plate-test")
RESULTS = ROOT / "results"
DETECTIONS = RESULTS / "plate_coordinates.json"
STATE = RESULTS / "anpr_processed_state.json"
MASTER = RESULTS / "anpr_results.json"
BATCH_INPUT = RESULTS / "anpr_pending.json"
BATCH_OUTPUT = RESULTS / "anpr_batch_output.json"

def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)

print("[INFO] ANPR worker started", flush=True)

while True:
    try:
        records = load_json(DETECTIONS, [])
        state = load_json(STATE, {})
        pending = []

        for record in records if isinstance(records, list) else []:
            image_value = record.get("image", "")
            detections = record.get("detections", [])
            if not image_value or not detections:
                continue

            image_path = Path(image_value)
            if not image_path.is_file():
                continue

            stat = image_path.stat()
            signature = f"{stat.st_size}:{stat.st_mtime_ns}"
            key = str(image_path)

            if state.get(key) != signature:
                pending.append(record)

        if pending:
            BATCH_INPUT.write_text(
                json.dumps(pending, indent=2), encoding="utf-8"
            )
            BATCH_OUTPUT.unlink(missing_ok=True)

            cmd = [
                str(ROOT / "venv/bin/python"),
                str(ROOT / "anpr_test.py"),
                "--detections_json", str(BATCH_INPUT),
                "--device", "cpu",
                "--output_json", str(BATCH_OUTPUT),
            ]
            print(f"[INFO] Sending {len(pending)} image(s) to OCR", flush=True)
            run = subprocess.run(cmd, cwd=ROOT)

            if run.returncode == 0 and BATCH_OUTPUT.is_file():
                batch_results = load_json(BATCH_OUTPUT, None)
                if isinstance(batch_results, list):
                    existing = load_json(MASTER, [])
                    if not isinstance(existing, list):
                        existing = []
                    existing.extend(batch_results)
                    save_json(MASTER, existing)

                    for record in pending:
                        old_path = Path(record["image"])
                        new_path = old_path
                        for item in batch_results:
                            if item.get("image") == old_path.name:
                                prediction = item.get("prediction", "")
                                confidence = item.get("ocr_confidence", 0)
                                if prediction and confidence >= 0.90:
                                    candidate = old_path.with_name(
                                        prediction + old_path.suffix.lower()
                                    )
                                    if candidate.is_file():
                                        new_path = candidate
                                        break

                        if new_path.is_file():
                            stat = new_path.stat()
                            state[str(new_path)] = (
                                f"{stat.st_size}:{stat.st_mtime_ns}"
                            )
                        else:
                            state[str(old_path)] = "processed"

                    save_json(STATE, state)
                    print(f"[INFO] OCR batch completed: {len(batch_results)} result(s)", flush=True)

    except Exception as exc:
        print(f"[ERROR] ANPR worker: {exc}", flush=True)

    time.sleep(2)
