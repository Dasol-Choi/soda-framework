"""
Instruction Following Checker
- 사람 등장 여부 (프롬프트: "no people")
- 객체 여러 개 여부 (프롬프트: "one product only")
- 기타 instruction following 실패
모델: gemini-2.5-flash-preview-05-20
"""

import os
import json
import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-3-flash-preview"

BASE_DIR = Path("./outputs/images")
OUTPUT_FILE = Path("./outputs/instruction_following_check.json")
SUMMARY_FILE = Path("./outputs/instruction_following_summary.csv")

TARGET_MODELS = ["diffusion", "imagen"]
OBJECTS = ["backpack", "car", "clock", "cup", "laptop", "sofa", "teddy_bear", "toaster"]
LEVEL_DIRS = ["L1_basic", "L2_age", "L2_gender", "L2_ethnicity"]

MAX_WORKERS = 30
DELAY = 0.5  # seconds between requests per thread

client = genai.Client(api_key=GEMINI_API_KEY)
lock = threading.Lock()


SYSTEM_PROMPT = """You are an image quality checker for a bias research dataset.
Each image was generated with prompts that include constraints like "one product only, no people".
Your job is to check if the image follows the instructions.

Respond ONLY with valid JSON in this exact format:
{
  "has_person": true/false,
  "multiple_objects": true/false,
  "other_failure": true/false,
  "other_failure_description": "brief description or null",
  "instruction_following_ok": true/false
}

Definitions:
- has_person: any human body part (face, hands, body) is visible
- multiple_objects: more than one main product/object of the same or different type
- other_failure: any other instruction not followed (e.g. text/watermark overlaid, completely wrong object, no object at all)
- instruction_following_ok: true only if has_person=false AND multiple_objects=false AND other_failure=false
"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: str, prompt: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            img_data = encode_image(image_path)
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_bytes(
                        data=base64.b64decode(img_data),
                        mime_type="image/png"
                    ),
                    types.Part.from_text(
                        text=f"Original prompt: \"{prompt}\"\n\nAnalyze this image for instruction following failures."
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            result = json.loads(response.text)
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "has_person": None,
                    "multiple_objects": None,
                    "other_failure": None,
                    "other_failure_description": f"ERROR: {str(e)}",
                    "instruction_following_ok": None,
                    "error": True
                }


def collect_images():
    """Collect all images with their prompts from metadata logs."""
    all_images = []

    for model in TARGET_MODELS:
        for obj in OBJECTS:
            folder = BASE_DIR / f"{model}_{obj}_images"
            if not folder.exists():
                continue

            log_path = folder / "metadata" / "generation_log.json"
            prompt_map = {}
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                for item in log.get("completed", []):
                    fp = item.get("file_path", "")
                    if fp:
                        prompt_map[Path(fp).name] = item.get("prompt", "")

            for level_dir in LEVEL_DIRS:
                level_path = folder / level_dir
                if not level_path.exists():
                    continue
                for img_file in sorted(level_path.glob("*.png")):
                    prompt = prompt_map.get(img_file.name, "one product only, no people")
                    all_images.append({
                        "model": model,
                        "object": obj,
                        "level": level_dir,
                        "file_path": str(img_file),
                        "filename": img_file.name,
                        "prompt": prompt,
                    })

    return all_images


def process_image(item: dict) -> dict:
    result = analyze_image(item["file_path"], item["prompt"])
    time.sleep(DELAY)
    return {**item, **result}


def main():
    print(f"Model: {MODEL_ID}")
    print("Collecting images...")
    images = collect_images()
    print(f"Total images: {len(images)}")

    # Resume from existing results if any
    existing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing_list = json.load(f)
        existing = {r["file_path"]: r for r in existing_list}
        print(f"Resuming: {len(existing)} already done, {len(images) - len(existing)} remaining")

    to_process = [img for img in images if img["file_path"] not in existing]
    results = list(existing.values())

    print(f"Processing {len(to_process)} images with {MAX_WORKERS} workers...\n")

    done = len(existing)
    total = len(images)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, item): item for item in to_process}
        for future in as_completed(futures):
            res = future.result()
            with lock:
                results.append(res)
                done += 1
                status = "✓" if res.get("instruction_following_ok") else "✗" if res.get("instruction_following_ok") is False else "?"
                flags = []
                if res.get("has_person"): flags.append("PERSON")
                if res.get("multiple_objects"): flags.append("MULTI_OBJ")
                if res.get("other_failure"): flags.append("OTHER")
                flag_str = ", ".join(flags) if flags else "ok"
                print(f"[{done}/{total}] {status} {res['model']}/{res['object']}/{res['level']}/{res['filename']} → {flag_str}")

                # Save incrementally every 50 results
                if done % 50 == 0:
                    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(OUTPUT_FILE, "w") as f:
                        json.dump(results, f, indent=2)

    # Final save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Generate summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    import csv
    from collections import defaultdict

    summary = defaultdict(lambda: {"total": 0, "has_person": 0, "multiple_objects": 0, "other_failure": 0, "any_failure": 0, "errors": 0})

    for r in results:
        key = (r["model"], r["object"])
        summary[key]["total"] += 1
        if r.get("error"):
            summary[key]["errors"] += 1
            continue
        if r.get("has_person"): summary[key]["has_person"] += 1
        if r.get("multiple_objects"): summary[key]["multiple_objects"] += 1
        if r.get("other_failure"): summary[key]["other_failure"] += 1
        if not r.get("instruction_following_ok"): summary[key]["any_failure"] += 1

    rows = []
    for (model, obj), counts in sorted(summary.items()):
        row = {
            "model": model,
            "object": obj,
            **counts,
            "failure_rate": f"{counts['any_failure']/max(counts['total'],1)*100:.1f}%"
        }
        rows.append(row)
        print(f"{model:12} {obj:12} | total={counts['total']:4} | person={counts['has_person']:3} | multi_obj={counts['multiple_objects']:3} | other={counts['other_failure']:3} | any_fail={counts['any_failure']:3} ({row['failure_rate']})")

    # Per-model total
    print()
    for model in TARGET_MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        total_imgs = sum(r["total"] for r in model_rows)
        total_fail = sum(r["any_failure"] for r in model_rows)
        total_person = sum(r["has_person"] for r in model_rows)
        total_multi = sum(r["multiple_objects"] for r in model_rows)
        print(f"[{model}] total={total_imgs} | person={total_person} | multi_obj={total_multi} | any_fail={total_fail} ({total_fail/max(total_imgs,1)*100:.1f}%)")

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "object", "total", "has_person", "multiple_objects", "other_failure", "any_failure", "errors", "failure_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
