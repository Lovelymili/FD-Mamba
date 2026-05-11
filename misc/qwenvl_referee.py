
import os
import io
import re
import json
import time
import base64
import hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from openai import OpenAI


def _sha1(s: str) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().float().cpu()
    mx = float(img.max())
    mn = float(img.min())
    if mx <= 1.5 and mn >= 0.0:
        img = img * 255.0
    img = img.clamp(0.0, 255.0).byte()
    return img.permute(1, 2, 0).contiguous().numpy()


def _pil_to_data_url(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


class QwenVLRefereePostprocessor:
    """
    Qwen3-VL-Plus 组件级裁判（像素级修改仍由你 CD mask 执行）：
      - 从 pred mask 取 connected components
      - 对低置信度&可疑组件生成 3-panel: [pre | post | post+mask_overlay]
      - 调 Qwen3-VL-Plus 进行多分类判别（REAL_CHANGE/SHADOW/SEASONAL/MISREG/OTHER）
      - 映射到像素动作：drop / shrink / keep
      - cache 每个 component 的裁判结果，避免重复 API 调用
    """

    def __init__(
        self,
        model_name: str = "qwen3-vl-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY",
        device: Optional[torch.device] = None,

        thr: float = 0.5,
        topk: int = 12,
        pad: int = 12,
        patch_size: int = 192,
        cache_dir: str = "qwen_cache",

        
        mean_p_lo: float = 0.45,
        mean_p_hi: float = 0.75,
        p90_hi: float = 0.92,
        min_area: int = 32,           
        max_query_per_image: int = 12,

        
        drop_labels: Tuple[str, ...] = ("SEASONAL", "MISREGISTRATION"),  
        shadow_shrink_px: int = 1,   
        safety_min_keep_ratio: float = 0.60,  
        temperature: float = 0.0,
        max_tokens: int = 128,

        
        prompt: str = (
            "The image has three panels: LEFT=BEFORE, MIDDLE=AFTER, RIGHT=AFTER with a red highlighted region.\n"
            "Task: classify the red region.\n"
            "Choose EXACTLY ONE label from: [REAL_CHANGE, SHADOW, SEASONAL, MISREGISTRATION, OTHER]\n"
            "Rules:\n"
            "- REAL_CHANGE: real man-made change (building/road construction or demolition)\n"
            "- SHADOW: mostly shadow/illumination difference\n"
            "- SEASONAL: mostly seasonal vegetation/water appearance change\n"
            "- MISREGISTRATION: mostly misalignment/registration artifacts\n"
            "- OTHER: none of the above\n"
            "Output JSON ONLY, in the form: {\"label\":\"...\",\"reason\":\"...\"}\n"
        ),
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.device = device

        self.thr = float(thr)
        self.topk = int(topk)
        self.pad = int(pad)
        self.patch_size = int(patch_size)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.mean_p_lo = float(mean_p_lo)
        self.mean_p_hi = float(mean_p_hi)
        self.p90_hi = float(p90_hi)
        self.min_area = int(min_area)
        self.max_query_per_image = int(max_query_per_image)

        self.drop_labels = tuple(drop_labels)
        self.shadow_shrink_px = int(shadow_shrink_px)
        self.safety_min_keep_ratio = float(safety_min_keep_ratio)

        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.prompt = str(prompt)

        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key env var: {self.api_key_env}. "
                f"Please set: export {self.api_key_env}=YOUR_KEY"
            )

        
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    
    def _cc_from_mask(self, mask_hw_u8: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hw_u8, connectivity=8)
        comps = []
        for cid in range(1, num):
            x, y, w, h, area = stats[cid].tolist()
            comps.append({"id": int(cid), "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
        return labels.astype(np.int32), comps

    def _crop_with_pad(self, img: np.ndarray, x: int, y: int, w: int, h: int, pad: int) -> np.ndarray:
        H, W, _ = img.shape
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        return img[y0:y1, x0:x1, :]

    def _overlay_red(self, post_rgb_u8: np.ndarray, comp_mask_hw_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        
        post = post_rgb_u8.astype(np.float32)
        red = np.zeros_like(post)
        red[..., 0] = 255.0
        a = (comp_mask_hw_u8.astype(np.float32) * alpha)[..., None]  
        out = post * (1 - a) + red * a
        return out.clip(0, 255).astype(np.uint8)

    def _make_triplet(self, pre_rgb: np.ndarray, post_rgb: np.ndarray, labels_hw: np.ndarray, cid: int) -> Image.Image:
        comp = (labels_hw == cid).astype(np.uint8)
        ys, xs = np.where(comp > 0)
        if ys.size == 0:
            return Image.new("RGB", (self.patch_size * 3, self.patch_size), (0, 0, 0))

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1

        pre_c = self._crop_with_pad(pre_rgb, x0, y0, w, h, self.pad)
        post_c = self._crop_with_pad(post_rgb, x0, y0, w, h, self.pad)

        
        H, W, _ = post_rgb.shape
        x0p = max(0, x0 - self.pad)
        y0p = max(0, y0 - self.pad)
        x1p = min(W, x0 + w + self.pad)
        y1p = min(H, y0 + h + self.pad)
        comp_c = comp[y0p:y1p, x0p:x1p]

        post_ov = self._overlay_red(post_c, comp_c, alpha=0.45)

        ps = self.patch_size
        a = Image.fromarray(pre_c, mode="RGB").resize((ps, ps))
        b = Image.fromarray(post_c, mode="RGB").resize((ps, ps))
        c = Image.fromarray(post_ov, mode="RGB").resize((ps, ps))

        row = Image.new("RGB", (ps * 3, ps), (0, 0, 0))
        row.paste(a, (0, 0))
        row.paste(b, (ps, 0))
        row.paste(c, (ps * 2, 0))
        return row

    
    def _parse_json(self, s: str) -> Tuple[str, str]:
        """
        期望模型输出 JSON：{"label":"...","reason":"..."}
        容错：提取第一个 {...} 块
        """
        if s is None:
            return "OTHER", ""
        txt = str(s).strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            txt = m.group(0)
        try:
            obj = json.loads(txt)
            label = str(obj.get("label", "OTHER")).strip().upper()
            reason = str(obj.get("reason", "")).strip()
            if label not in {"REAL_CHANGE", "SHADOW", "SEASONAL", "MISREGISTRATION", "OTHER"}:
                label = "OTHER"
            return label, reason
        except Exception:
            
            up = txt.upper()
            for k in ["REAL_CHANGE", "SHADOW", "SEASONAL", "MISREGISTRATION", "OTHER"]:
                if k in up:
                    return k, txt[:200]
            return "OTHER", txt[:200]

    def _judge_one(self, pil_img: Image.Image) -> Tuple[str, str]:
        img_url = _pil_to_data_url(pil_img)

        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
        )
        out = resp.choices[0].message.content
        return self._parse_json(out)

    
    def refine_one(self, pre_chw: torch.Tensor, post_chw: torch.Tensor, logits_chw: torch.Tensor, sample_id: str) -> torch.Tensor:
        pre = _to_uint8_rgb(pre_chw)
        post = _to_uint8_rgb(post_chw)

        prob = F.softmax(logits_chw.detach().float(), dim=0)[1].cpu().numpy()
        orig = (prob > self.thr).astype(np.uint8)

        labels, comps = self._cc_from_mask(orig)
        if len(comps) == 0:
            return torch.from_numpy(orig).long()

        
        candidates = []
        for c in comps:
            cid = int(c["id"])
            area = int(c["area"])
            if area < self.min_area:
                continue
            m = (labels == cid)
            if m.sum() == 0:
                continue
            mean_p = float(prob[m].mean())
            p90 = float(np.quantile(prob[m], 0.90))
            c["mean_p"] = mean_p
            c["p90"] = p90
            if (self.mean_p_lo <= mean_p <= self.mean_p_hi) and (p90 <= self.p90_hi):
                candidates.append(c)

        if len(candidates) == 0:
            return torch.from_numpy(orig).long()

        candidates = sorted(candidates, key=lambda x: int(x["area"]), reverse=True)[: min(self.topk, self.max_query_per_image)]
        refined = orig.copy()

        for c in candidates:
            cid = int(c["id"])
            cache_key = _sha1(
                f"{sample_id}|cid={cid}|thr={self.thr:.3f}|topk={self.topk}|pad={self.pad}|patch={self.patch_size}"
                f"|mean={self.mean_p_lo:.2f}-{self.mean_p_hi:.2f}|p90hi={self.p90_hi:.2f}"
                f"|prompt={self.prompt}"
                f"|model={self.model_name}|base={self.base_url}"
            )
            cache_path = os.path.join(self.cache_dir, cache_key + ".json")

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    label = str(obj.get("label", "OTHER")).upper()
                except Exception:
                    label = "OTHER"
            else:
                trip = self._make_triplet(pre, post, labels, cid)
                label, reason = self._judge_one(trip)
                obj = {"label": label, "reason": reason, "ts": time.time()}
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False)
                except Exception:
                    pass

            
            if label in self.drop_labels:
                refined[labels == cid] = 0
            elif label == "SHADOW" and self.shadow_shrink_px > 0:
                
                comp = (labels == cid).astype(np.uint8)
                k = 2 * self.shadow_shrink_px + 1
                kernel = np.ones((k, k), np.uint8)
                comp2 = cv2.erode(comp, kernel, iterations=1)
                
                refined[labels == cid] = 0
                refined[comp2 > 0] = 1
            else:
                
                pass

        
        orig_sum = float(orig.sum())
        if orig_sum > 0 and refined.sum() < self.safety_min_keep_ratio * orig_sum:
            refined = orig

        return torch.from_numpy(refined.astype(np.uint8)).long()

    def refine_batch(self, img1_bchw: torch.Tensor, img2_bchw: torch.Tensor, logits_bchw: torch.Tensor, sample_ids: List[str]) -> torch.Tensor:
        outs = []
        b = int(img1_bchw.shape[0])
        for i in range(b):
            outs.append(self.refine_one(img1_bchw[i], img2_bchw[i], logits_bchw[i], sample_ids[i]))
        return torch.stack(outs, dim=0)
