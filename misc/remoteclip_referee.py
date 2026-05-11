import os
import re
import json
import time
import hashlib
import tempfile
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


from gradio_client import Client


def _to_uint8_rgb(img_chw: torch.Tensor) -> np.ndarray:
    img = img_chw.detach().float().cpu()
    mx = float(img.max())
    mn = float(img.min())
    if mx <= 1.5 and mn >= 0.0:
        img = img * 255.0
    img = img.clamp(0.0, 255.0).byte()
    return img.permute(1, 2, 0).contiguous().numpy()


def _sha1(s: str) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8"))
    return h.hexdigest()





class RemoteCLIPRefereePostprocessor:
    """
    用 RemoteCLIP 做“组件级”裁判：
      - 从预测 mask 得到 connected components
      - 每个 component crop 出 (pre|post) 拼接图
      - 用 RemoteCLIP 对两条文本提示做相似度比较：keep / drop
      - 输出 refined mask
    """

    def __init__(
        self,
        model,
        preprocess,
        tokenizer,
        device: torch.device,
        thr: float = 0.5,
        topk: int = 12,
        pad: int = 12,
        patch_size: int = 192,
        cache_dir: str = "rclip_cache",
        prompt_keep: str = "a real land cover change between two satellite images",
        prompt_drop: str = "no real change, only lighting/shadow difference, seasonal change, or misregistration",
        margin_keep: float = 0.02,
        auto_shrink_px: int = 0,
        drop_margin: float = 0.55,
    ):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device

        self.thr = float(thr)
        self.topk = int(topk)
        self.pad = int(pad)
        self.patch_size = int(patch_size)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.prompt_keep = str(prompt_keep)
        self.prompt_drop = str(prompt_drop)
        self.margin_keep = float(margin_keep)
        self.auto_shrink_px = int(auto_shrink_px)
        self.drop_margin = float(drop_margin)

        
        with torch.no_grad():
            text = self.tokenizer([self.prompt_keep, self.prompt_drop])
            if isinstance(text, torch.Tensor):
                text = text.to(self.device)
            else:
                text = torch.tensor(text).to(self.device)

            self.text_features = self.model.encode_text(text)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def _cc_from_mask(self, mask_hw_u8: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hw_u8, connectivity=8)
        comps = []
        for cid in range(1, num):
            x, y, w, h, area = stats[cid].tolist()
            comps.append({"id": int(cid), "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
        return labels.astype(np.int32), comps

    def _crop_with_pad(self, img: np.ndarray, x: int, y: int, w: int, h: int, pad: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        H, W, _ = img.shape
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        return img[y0:y1, x0:x1, :], (x0, y0, x1, y1)

    def _make_pair(self, pre_rgb: np.ndarray, post_rgb: np.ndarray, labels_hw: np.ndarray, cid: int) -> Image.Image:
        comp_mask = (labels_hw == cid).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1

        pre_c, _ = self._crop_with_pad(pre_rgb, x0, y0, w, h, self.pad)
        post_c, _ = self._crop_with_pad(post_rgb, x0, y0, w, h, self.pad)

        ps = self.patch_size
        a = Image.fromarray(pre_c, mode="RGB").resize((ps, ps))
        b = Image.fromarray(post_c, mode="RGB").resize((ps, ps))

        row = Image.new("RGB", (ps * 2, ps), (0, 0, 0))
        row.paste(a, (0, 0))
        row.paste(b, (ps, 0))
        return row

    def _score_keep_drop(self, pil_img: Image.Image) -> Tuple[float, float]:
        img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            img_f = self.model.encode_image(img_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            logits = (100.0 * img_f @ self.text_features.T).squeeze(0)
        return float(logits[0].item()), float(logits[1].item())

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
            m = (labels == cid)
            if m.sum() == 0:
                continue
            mean_p = float(prob[m].mean())
            p90 = float(np.quantile(prob[m], 0.90))
            c["mean_p"] = mean_p
            c["p90"] = p90
            if (0.45 <= mean_p <= 0.75) and (p90 <= 0.92):
                candidates.append(c)

        if len(candidates) == 0:
            return torch.from_numpy(orig).long()

        candidates = sorted(candidates, key=lambda x: int(x["area"]), reverse=True)[: self.topk]
        keep_ids = [int(c["id"]) for c in candidates]

        refined = orig.copy()
        drop_margin = float(self.drop_margin)

        for cid in keep_ids:
            mode = "pair"
            cache_key = _sha1(
                f"{sample_id}|mode={mode}|thr={self.thr:.3f}|cid={cid}|topk={self.topk}"
                f"|pad={self.pad}|patch={self.patch_size}|dropm={drop_margin:.3f}"
                f"|keep={self.prompt_keep}|drop={self.prompt_drop}"
            )
            cache_path = os.path.join(self.cache_dir, cache_key + ".txt")

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        line = f.read().strip()
                    s_keep, s_drop = [float(x) for x in line.split(",")]
                except Exception:
                    pair = self._make_pair(pre, post, labels, cid)
                    s_keep, s_drop = self._score_keep_drop(pair)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(f"{s_keep},{s_drop}")
            else:
                pair = self._make_pair(pre, post, labels, cid)
                s_keep, s_drop = self._score_keep_drop(pair)
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(f"{s_keep},{s_drop}")

            diff = float(s_keep - s_drop)
            decision_keep = True
            if diff < -drop_margin:
                decision_keep = False

            if not decision_keep:
                refined[labels == cid] = 0

        
        orig_sum = float(orig.sum())
        if orig_sum > 0 and refined.sum() < 0.60 * orig_sum:
            refined = orig

        if self.auto_shrink_px > 0:
            k = 2 * self.auto_shrink_px + 1
            kernel = np.ones((k, k), np.uint8)
            refined = cv2.erode(refined, kernel, iterations=1)

        return torch.from_numpy(refined.astype(np.uint8)).long()

    def refine_batch(self, img1_bchw: torch.Tensor, img2_bchw: torch.Tensor, logits_bchw: torch.Tensor, sample_ids: List[str]) -> torch.Tensor:
        outs = []
        b = int(img1_bchw.shape[0])
        for i in range(b):
            outs.append(self.refine_one(img1_bchw[i], img2_bchw[i], logits_bchw[i], sample_ids[i]))
        return torch.stack(outs, dim=0)





class GeoPixGradioJudge:
    """
    GeoPix 本地 gradio demo 裁判：
      predict(task_type, input_str, input_image, api_name="/inference") -> (text_output, image_output)
    """

    def __init__(self, server_url: str = "http://127.0.0.1:7860/", api_name: str = "/inference"):
        self.client = Client(server_url)
        self.api_name = api_name

    @staticmethod
    def _parse_yes_no(text: str) -> Optional[bool]:
        if text is None:
            return None
        t = str(text).strip().upper()
        if re.search(r"\bYES\b", t):
            return True
        if re.search(r"\bNO\b", t):
            return False
        return None

    def judge_keep(self, pair_img: Image.Image, question: str, default_keep: bool = True) -> Tuple[bool, str]:
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            pair_img.save(tmp_path)

            text_output, _image_output = self.client.predict(
                "Visual Question Answering",
                question,
                tmp_path,
                api_name=self.api_name,
            )
            parsed = self._parse_yes_no(text_output)
            if parsed is None:
                return default_keep, str(text_output)
            return parsed, str(text_output)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class GeoPixRefereePostprocessor:
    """
    用 GeoPix (via gradio demo) 做组件级裁判：
      - 与 RemoteCLIP 版本复用同一套候选筛选/连通域/安全阀
      - 对每个候选 component，把 pre|post crop 拼接成一张图喂给 GeoPix VQA
      - GeoPix 输出 YES/NO，决定 keep/drop
      - cache 每个 component 的裁判结果（否则太慢）
    """

    def __init__(
        self,
        geopix_url: str = "http://127.0.0.1:7860/",
        geopix_api_name: str = "/inference",
        thr: float = 0.5,
        topk: int = 12,
        pad: int = 12,
        patch_size: int = 192,
        cache_dir: str = "geopix_cache",
        auto_shrink_px: int = 0,
        
        mean_p_lo: float = 0.45,
        mean_p_hi: float = 0.75,
        p90_hi: float = 0.92,
        
        min_keep_ratio: float = 0.60,
        
        question: str = (
            "Compare the LEFT (before) and RIGHT (after) satellite images in the provided patch. "
            "Is there a REAL land-cover change (e.g., building/road construction or demolition), "
            "rather than shadow/illumination difference, seasonal vegetation change, or misregistration? "
            "Answer with ONLY one token: YES or NO."
        ),
    ):
        self.judge = GeoPixGradioJudge(geopix_url, geopix_api_name)

        self.thr = float(thr)
        self.topk = int(topk)
        self.pad = int(pad)
        self.patch_size = int(patch_size)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.auto_shrink_px = int(auto_shrink_px)

        self.mean_p_lo = float(mean_p_lo)
        self.mean_p_hi = float(mean_p_hi)
        self.p90_hi = float(p90_hi)

        self.min_keep_ratio = float(min_keep_ratio)
        self.question = str(question)

    def _cc_from_mask(self, mask_hw_u8: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hw_u8, connectivity=8)
        comps = []
        for cid in range(1, num):
            x, y, w, h, area = stats[cid].tolist()
            comps.append({"id": int(cid), "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
        return labels.astype(np.int32), comps

    def _crop_with_pad(self, img: np.ndarray, x: int, y: int, w: int, h: int, pad: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        H, W, _ = img.shape
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        return img[y0:y1, x0:x1, :], (x0, y0, x1, y1)

    def _make_pair(self, pre_rgb: np.ndarray, post_rgb: np.ndarray, labels_hw: np.ndarray, cid: int) -> Image.Image:
        comp_mask = (labels_hw == cid).astype(np.uint8)
        ys, xs = np.where(comp_mask > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1

        pre_c, _ = self._crop_with_pad(pre_rgb, x0, y0, w, h, self.pad)
        post_c, _ = self._crop_with_pad(post_rgb, x0, y0, w, h, self.pad)

        ps = self.patch_size
        a = Image.fromarray(pre_c, mode="RGB").resize((ps, ps))
        b = Image.fromarray(post_c, mode="RGB").resize((ps, ps))

        row = Image.new("RGB", (ps * 2, ps), (0, 0, 0))
        row.paste(a, (0, 0))
        row.paste(b, (ps, 0))
        return row

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

        candidates = sorted(candidates, key=lambda x: int(x["area"]), reverse=True)[: self.topk]
        keep_ids = [int(c["id"]) for c in candidates]

        refined = orig.copy()

        for cid in keep_ids:
            mode = "pair_vqa"
            cache_key = _sha1(
                f"{sample_id}|mode={mode}|thr={self.thr:.3f}|cid={cid}|topk={self.topk}"
                f"|pad={self.pad}|patch={self.patch_size}"
                f"|q={self.question}"
                f"|meanlo={self.mean_p_lo:.2f}|meanhi={self.mean_p_hi:.2f}|p90hi={self.p90_hi:.2f}"
            )
            cache_path = os.path.join(self.cache_dir, cache_key + ".json")

            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    keep = bool(obj.get("keep", True))
                except Exception:
                    keep = True
            else:
                pair = self._make_pair(pre, post, labels, cid)
                keep, ans = self.judge.judge_keep(pair, self.question, default_keep=True)
                obj = {"keep": bool(keep), "answer": str(ans), "ts": time.time()}
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(obj, f)
                except Exception:
                    pass

            if not keep:
                refined[labels == cid] = 0

        
        orig_sum = float(orig.sum())
        if orig_sum > 0 and refined.sum() < float(self.min_keep_ratio) * orig_sum:
            refined = orig

        if self.auto_shrink_px > 0:
            k = 2 * self.auto_shrink_px + 1
            kernel = np.ones((k, k), np.uint8)
            refined = cv2.erode(refined, kernel, iterations=1)

        return torch.from_numpy(refined.astype(np.uint8)).long()

    def refine_batch(self, img1_bchw: torch.Tensor, img2_bchw: torch.Tensor, logits_bchw: torch.Tensor, sample_ids: List[str]) -> torch.Tensor:
        outs = []
        b = int(img1_bchw.shape[0])
        for i in range(b):
            outs.append(self.refine_one(img1_bchw[i], img2_bchw[i], logits_bchw[i], sample_ids[i]))
        return torch.stack(outs, dim=0)
