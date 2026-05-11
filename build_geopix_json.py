#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate GeoPix-like JSON annotations for CD datasets (A/B pair) using:
  - OpenAI-compatible vision endpoint (/v1/chat/completions), OR
  - Gemini generateContent endpoint (/v1beta/models/{model}:generateContent)

Outputs:
  <out_dir>/<split>.jsonl            # each line: {"id", "split", "img_name", "A","B","L","geopix": {...}}
  <out_dir>/<split>.errors.jsonl     # failures
  <out_dir>/index.json               # id -> geopix

Enhancements:
  - Skip if id already exists in <split>.jsonl
  - Multithreaded LLM requests (ThreadPoolExecutor)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed





CHANGE_TYPES = [
    "new_building", "demolished_building",
    "road_constructed", "road_removed",
    "bare_to_vegetation", "vegetation_to_bare",
    "water_change", "other_manmade",
    "no_change", "uncertain"
]
NUISANCE_TYPES = ["shadow", "illumination", "season", "misalignment", "cloud", "sensor_noise"]
SCENE_TYPES = ["urban", "suburban", "rural", "forest", "farmland", "water", "mixed", "unknown"]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_img_name_list(list_path: Path) -> List[str]:
    names: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.append(s.split()[0])
    return names


def get_paths(root_dir: Path, img_name: str) -> Tuple[Path, Path, Path]:
    return root_dir / "A" / img_name, root_dir / "B" / img_name, root_dir / "label" / img_name


def make_pair_image(a_path: Path, b_path: Path, out_path: Path, resize_to: Optional[int] = 512) -> None:
    with Image.open(a_path).convert("RGB") as ia, Image.open(b_path).convert("RGB") as ib:
        if resize_to is not None:
            ia = ia.resize((resize_to, resize_to), Image.BILINEAR)
            ib = ib.resize((resize_to, resize_to), Image.BILINEAR)
        w, h = ia.size
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(ia, (0, 0))
        canvas.paste(ib, (w, 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=92)


def image_to_data_url(img_path: Path) -> str:
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suf = img_path.suffix.lower()
    mime = "image/jpeg" if suf in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"


def prompt_json_only() -> str:
    return (
        "The image contains BEFORE (left) and AFTER (right) satellite views.\n"
        "Return ONLY ONE JSON object. No markdown. No explanation.\n"
        "Keys exactly: scene, change_types, nuisance, confidence, notes.\n"
        f"- scene: one of {SCENE_TYPES}\n"
        f"- change_types: up to 3 from {CHANGE_TYPES}\n"
        f"- nuisance: from {NUISANCE_TYPES}\n"
        "- confidence: map each chosen label to a float in [0,1]\n"
        "- notes: <= 20 words\n"
        "Example:\n"
        "{\"scene\":\"urban\",\"change_types\":[\"new_building\"],\"nuisance\":[\"shadow\"],"
        "\"confidence\":{\"new_building\":0.7,\"shadow\":0.4},\"notes\":\"construction near roads\"}\n"
        "Now output JSON:"
    )


def _extract_braced_block(text: str) -> Optional[str]:
    text = text.strip()
    text = text.replace("```json", "```").replace("```JSON", "```")
    if "```" in text:
        parts = text.split("```")
        cand = None
        for p in parts:
            if "{" in p and "}" in p:
                if cand is None or len(p) > len(cand):
                    cand = p
        if cand is not None:
            text = cand.strip()
    m = _JSON_RE.search(text)
    if not m:
        return None
    return m.group(0)


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    block = _extract_braced_block(text)
    if block is None:
        return None

    
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    
    fixed = block.replace("'", "\"")
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    scene = str(obj.get("scene", "unknown")).strip().lower()
    if scene not in SCENE_TYPES:
        scene = "unknown"
    out["scene"] = scene

    change_types = obj.get("change_types", [])
    if isinstance(change_types, str):
        change_types = [change_types]
    change_types = [str(x).strip() for x in change_types if str(x).strip()]
    change_types = [ct for ct in change_types if ct in CHANGE_TYPES]
    if not change_types:
        change_types = ["uncertain"]
    out["change_types"] = change_types[:3]

    nuisance = obj.get("nuisance", [])
    if isinstance(nuisance, str):
        nuisance = [nuisance]
    nuisance = [str(x).strip().lower() for x in nuisance if str(x).strip()]
    nuisance = [n for n in nuisance if n in NUISANCE_TYPES]
    out["nuisance"] = nuisance[:6]

    conf = obj.get("confidence", {})
    if not isinstance(conf, dict):
        conf = {}
    conf_out: Dict[str, float] = {}
    for k, v in conf.items():
        kk = str(k).strip()
        try:
            vv = float(v)
        except Exception:
            continue
        vv = max(0.0, min(1.0, vv))
        conf_out[kk] = vv
    for ct in out["change_types"]:
        conf_out.setdefault(ct, 0.5)
    out["confidence"] = conf_out

    notes = str(obj.get("notes", "")).strip()
    if len(notes.split()) > 25:
        notes = " ".join(notes.split()[:25])
    out["notes"] = notes
    return out


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]





def openai_chat_vision(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_data_url: str,
    timeout_s: int = 180,
    temperature: float = 0.2,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a precise assistant. Output strict JSON only."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ],
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]





def gemini_generate_content(
    endpoint_base: str,   
    token: str,
    model: str,           
    prompt: str,
    image_path: str,      
    timeout_s: int = 180,
    max_retries: int = 6,
    temperature: float = 0.2,
) -> Tuple[str, Dict[str, Any]]:
    url = endpoint_base.rstrip("/") + f"/v1beta/models/{model}:generateContent"

    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime, "data": b64}},
            ],
        }],
        "generationConfig": {"temperature": float(temperature), "topP": 1.0},
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout_s)
            if r.status_code in (429, 502, 503, 504):
                sleep = min(60.0, (2 ** attempt) * 1.0) + random.uniform(0, 0.5)
                time.sleep(sleep)
                last_err = RuntimeError(f"{r.status_code} {r.text[:200]}")
                continue
            r.raise_for_status()
            j = r.json()
            parts = j.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
            return text, j
        except Exception as e:
            last_err = e
            time.sleep(min(60.0, (2 ** attempt) * 1.0))

    raise RuntimeError(f"Gemini generateContent failed after retries: {last_err}")





def load_done_ids(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                if isinstance(j, dict) and "id" in j:
                    done.add(str(j["id"]))
            except Exception:
                
                continue
    return done





def process_one_sample(
    base_rec: Dict[str, Any],
    backend: str,
    base_url: str,
    api_key: str,
    model: str,
    base_prompt: str,
    pair_path: Path,
    timeout_s: int,
    temperature: float,
    max_retry: int,
) -> Dict[str, Any]:
    """
    Returns:
      {"ok": True, "record": {...}} OR {"ok": False, "error_record": {...}}
    """
    try:
        obj: Optional[Dict[str, Any]] = None
        last_text: Optional[str] = None

        for t in range(max_retry + 1):
            cur_prompt = base_prompt if t == 0 else (
                base_prompt + "\nNO TEXT. JSON ONLY. START WITH { AND END WITH }."
            )

            if backend == "openai":
                data_url = image_to_data_url(pair_path)
                text_out = openai_chat_vision(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=cur_prompt,
                    image_data_url=data_url,
                    timeout_s=timeout_s,
                    temperature=temperature,
                )
            elif backend == "gemini":
                text_out, _ = gemini_generate_content(
                    endpoint_base=base_url,
                    token=api_key,
                    model=model,
                    prompt=cur_prompt,
                    image_path=str(pair_path),
                    timeout_s=timeout_s,
                    max_retries=max(2, min(6, max_retry)),  
                    temperature=temperature,
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

            last_text = text_out
            raw = try_parse_json(text_out)
            if raw is not None:
                obj = normalize_schema(raw)
                break
            time.sleep(0.25)

        if obj is None:
            h = short_hash((last_text or "")[:2000])
            raise ValueError(f"No JSON object found in response. last_hash={h}")

        return {"ok": True, "record": {**base_rec, "geopix": obj}}
    except Exception as e:
        return {"ok": False, "error_record": {**base_rec, "error": str(e)}}





def run_split(
    root_dir: Path,
    split: str,
    out_dir: Path,
    pair_cache: Path,
    backend: str,                 
    base_url: str,                
    api_key: str,                 
    model: str,
    resize_to: int,
    sleep_s: float,
    resume: bool,
    max_retry: int,
    timeout_s: int,
    temperature: float,
    num_workers: int,
) -> None:
    list_path = root_dir / "list" / f"{split}.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing list file: {list_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{split}.jsonl"
    out_err = out_dir / f"{split}.errors.jsonl"

    
    done_ids = load_done_ids(out_jsonl) if resume else set()

    names = load_img_name_list(list_path)
    base_prompt = prompt_json_only()

    
    write_lock = threading.Lock()
    done_lock = threading.Lock()

    
    fw = out_jsonl.open("a", encoding="utf-8")
    fe = out_err.open("a", encoding="utf-8")

    def submit_tasks(exe: ThreadPoolExecutor):
        futures = []
        for img_name in names:
            sample_id = Path(img_name).stem

            if resume:
                with done_lock:
                    if sample_id in done_ids:
                        continue

            a_path, b_path, l_path = get_paths(root_dir, img_name)
            base_rec = {
                "id": sample_id,
                "split": split,
                "img_name": img_name,
                "A": str(a_path),
                "B": str(b_path),
                "L": str(l_path),
            }

            if (not a_path.exists()) or (not b_path.exists()):
                
                with write_lock:
                    fe.write(json.dumps({**base_rec, "error": "A or B not found."}, ensure_ascii=False) + "\n")
                    fe.flush()
                continue

            
            pair_path = pair_cache / split / f"{sample_id}.jpg"
            if not pair_path.exists():
                
                try:
                    make_pair_image(a_path, b_path, pair_path, resize_to=resize_to)
                except Exception as e:
                    with write_lock:
                        fe.write(json.dumps({**base_rec, "error": f"pair_image_failed: {e}"}, ensure_ascii=False) + "\n")
                        fe.flush()
                    continue

            
            fut = exe.submit(
                process_one_sample,
                base_rec=base_rec,
                backend=backend,
                base_url=base_url,
                api_key=api_key,
                model=model,
                base_prompt=base_prompt,
                pair_path=pair_path,
                timeout_s=timeout_s,
                temperature=temperature,
                max_retry=max_retry,
            )
            futures.append((sample_id, fut))
        return futures

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as exe:
            futures = submit_tasks(exe)

            for sample_id, fut in futures:
                res = fut.result()

                
                if sleep_s > 0:
                    time.sleep(sleep_s)

                if res.get("ok"):
                    rec = res["record"]
                    
                    if resume:
                        with done_lock:
                            if sample_id in done_ids:
                                continue
                            done_ids.add(sample_id)

                    with write_lock:
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fw.flush()
                else:
                    err_rec = res["error_record"]
                    with write_lock:
                        fe.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                        fe.flush()
    finally:
        fw.close()
        fe.close()


def build_index(out_dir: Path, splits: List[str]) -> None:
    index: Dict[str, Dict[str, Any]] = {}
    for sp in splits:
        p = out_dir / f"{sp}.jsonl"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if isinstance(j, dict) and "id" in j and "geopix" in j:
                        index[str(j["id"])] = j["geopix"]
                except Exception:
                    continue
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--splits", type=str, default="val")
    ap.add_argument("--pair_cache", type=str, default="tmp/geopix_pair_cache")

    ap.add_argument("--backend", type=str, choices=["openai", "gemini"], required=True)
    ap.add_argument("--base_url", type=str, required=True, help="openai: proxy base; gemini: endpoint base")
    ap.add_argument("--api_key", type=str, required=True, help="Bearer token/key")
    ap.add_argument("--model", type=str, required=True, help="openai/gemini model name")

    ap.add_argument("--resize_to", type=int, default=512)
    ap.add_argument("--sleep_s", type=float, default=0.0, help="sleep after each completed sample (usually 0 for multithread)")
    ap.add_argument("--resume", action="store_true", help="if set, skip ids already in <split>.jsonl")
    ap.add_argument("--max_retry", type=int, default=2, help="JSON-parse retries per sample")
    ap.add_argument("--timeout_s", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--num_workers", type=int, default=8, help="number of threads for concurrent LLM requests")

    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (root_dir / "geopix_json")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pair_cache = Path(args.pair_cache)

    for sp in splits:
        run_split(
            root_dir=root_dir,
            split=sp,
            out_dir=out_dir,
            pair_cache=pair_cache,
            backend=args.backend,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            resize_to=args.resize_to,
            sleep_s=args.sleep_s,
            resume=args.resume,
            max_retry=args.max_retry,
            timeout_s=args.timeout_s,
            temperature=args.temperature,
            num_workers=args.num_workers,
        )

    build_index(out_dir, splits)
    print(f"[OK] saved to: {out_dir} (index.json ready)")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
