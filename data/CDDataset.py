"""
CD Dataset (+ GeoPix JSON side information)

Folder:
root_dir
├─A
├─B
├─label
└─list
   ├─train.txt
   ├─val.txt
   └─test.txt

Optional GeoPix outputs:
root_dir
└─json
   ├─train.jsonl
   ├─val.jsonl
   └─test.jsonl
"""

from __future__ import annotations
import os
import random
import cv2
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import data.util as Util





IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = "B"
LABEL_FOLDER_NAME = "label"
LIST_FOLDER_NAME = "list"


CHANGE_TYPES = [
    "new_building", "demolished_building",
    "road_constructed", "road_removed",
    "bare_to_vegetation", "vegetation_to_bare",
    "water_change", "other_manmade",
    "no_change", "uncertain"
]
NUISANCE_TYPES = ["shadow", "illumination", "season", "misalignment", "cloud", "sensor_noise"]
SCENE_TYPES = ["urban", "suburban", "rural", "forest", "farmland", "water", "mixed", "unknown"]

CHANGE2ID = {k: i for i, k in enumerate(CHANGE_TYPES)}
NUIS2ID = {k: i for i, k in enumerate(NUISANCE_TYPES)}
SCENE2ID = {k: i for i, k in enumerate(SCENE_TYPES)}



from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F





class V2GammaCorrection(v2.Transform):
    """同步 Gamma 校正：确保 A/B 图像应用相同的随机亮度偏移"""
    def __init__(self, gamma_range=(0.8, 1.2)):
        super().__init__()
        self.gamma_range = gamma_range

    def make_params(self, flat_inputs):
        
        return dict(gamma=random.uniform(*self.gamma_range))

    def _transform(self, inpt, params):
        
        if "gamma" in params and isinstance(inpt, torch.Tensor):
            if inpt.ndim == 3 and inpt.shape[0] == 3:
                return F.adjust_gamma(inpt, params['gamma'])
        return inpt

class V2CLAHE(v2.Transform):
    """直方图均衡化：仅对图像生效，跳过标签"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def _transform(self, inpt, params):
        if isinstance(inpt, torch.Tensor) and inpt.ndim == 3 and inpt.shape[0] == 3:
            
            img = (inpt.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return torch.from_numpy(img).permute(2, 0, 1).to(inpt.device).float() / 255.0
        return inpt

class V2RandomErase(v2.Transform):
    """同步随机擦除：在 A/B 相同位置进行遮挡模拟"""
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3)):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def make_params(self, flat_inputs):
        img = next(i for i in flat_inputs if isinstance(i, torch.Tensor))
        _, h, w = img.shape
        apply = random.random() < self.p
        if not apply: return dict(apply=False)

        area = h * w
        erase_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(*self.ratio)
        eh, ew = int(np.sqrt(erase_area * aspect_ratio)), int(np.sqrt(erase_area / aspect_ratio))

        if eh < h and ew < w:
            y1, x1 = random.randint(0, h - eh), random.randint(0, w - ew)
            return dict(apply=True, x1=x1, y1=y1, h=eh, w=ew)
        return dict(apply=False)

    def _transform(self, inpt, params):
        if params.get('apply') and isinstance(inpt, torch.Tensor) and inpt.ndim == 3 and inpt.shape[0] == 3:
            x, y, h, w = params['x1'], params['y1'], params['h'], params['w']
            inpt[:, y:y+h, x:x+w] = torch.rand(inpt.shape[0], h, w)
        return inpt




def _load_img_name_list(list_path: Path) -> List[str]:
    
    names: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            names.append(s.split()[0])
    return names


def _get_paths(root_dir: Path, img_name: str) -> Tuple[Path, Path, Path]:
    a = root_dir / IMG_FOLDER_NAME / img_name
    b = root_dir / IMG_POST_FOLDER_NAME / img_name
    l = root_dir / LABEL_FOLDER_NAME / img_name
    return a, b, l


def _stem(img_name: str) -> str:
    return Path(img_name).stem


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_geopix_index(
    root_dir: Path,
    geopix_dirname: str,
    split: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Preferred: index.json (fast)
    Fallback: split.jsonl (build dict)
    """
    gp_dir = root_dir / geopix_dirname
    if not gp_dir.exists():
        return {}

    index_path = gp_dir / "index.json"
    if index_path.exists():
        obj = _safe_read_json(index_path)
        return obj if isinstance(obj, dict) else {}

    jsonl_path = gp_dir / f"{split}.jsonl"
    if jsonl_path.exists():
        out: Dict[str, Dict[str, Any]] = {}
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    if isinstance(j, dict) and "id" in j and "geopix" in j:
                        out[str(j["id"])] = j["geopix"]
                except Exception:
                    continue
        return out

    return {}


def _geopix_to_tensors(geopix: Optional[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Convert geopix dict -> numeric features.
    - geo_scene_id: scalar long
    - geo_change_ids: [3] long (padded with 'uncertain')
    - geo_nuisance_vec: [len(NUISANCE_TYPES)] float in [0,1]
    - geo_reliability: scalar float in [0,1] (heuristic gate, not "truth")
    """
    
    scene_id = torch.tensor(SCENE2ID["unknown"], dtype=torch.long)
    change_ids = torch.tensor([CHANGE2ID["uncertain"]] * 3, dtype=torch.long)
    nuis_vec = torch.zeros(len(NUISANCE_TYPES), dtype=torch.float32)
    reliability = torch.tensor(0.0, dtype=torch.float32)

    if not geopix:
        return {
            "geo_scene_id": scene_id,
            "geo_change_ids": change_ids,
            "geo_nuisance_vec": nuis_vec,
            "geo_reliability": reliability,
        }

    
    scene = str(geopix.get("scene", "unknown")).strip().lower()
    scene_id = torch.tensor(SCENE2ID.get(scene, SCENE2ID["unknown"]), dtype=torch.long)

    
    cts = geopix.get("change_types", [])
    if isinstance(cts, str):
        cts = [cts]
    cts = [str(x).strip() for x in cts if str(x).strip()]
    cts = [x for x in cts if x in CHANGE2ID]
    if not cts:
        cts = ["uncertain"]
    cts = (cts + ["uncertain"] * 3)[:3]
    change_ids = torch.tensor([CHANGE2ID[x] for x in cts], dtype=torch.long)

    
    nuis = geopix.get("nuisance", [])
    if isinstance(nuis, str):
        nuis = [nuis]
    nuis = [str(x).strip().lower() for x in nuis if str(x).strip()]
    nuis = [x for x in nuis if x in NUIS2ID]

    conf = geopix.get("confidence", {})
    if not isinstance(conf, dict):
        conf = {}

    
    for n in nuis:
        v = conf.get(n, 1.0)
        try:
            v = float(v)
        except Exception:
            v = 1.0
        v = max(0.0, min(1.0, v))
        nuis_vec[NUIS2ID[n]] = max(nuis_vec[NUIS2ID[n]], v)

    
    
    
    
    rel = 0.6
    if "uncertain" in cts:
        rel = 0.1
    if "no_change" in cts:
        
        k = sum(1 for x in nuis if x in ("season", "illumination", "shadow", "misalignment"))
        rel = 0.35 - 0.08 * max(0, k - 1)  
        rel = max(0.05, rel)
    reliability = torch.tensor(float(rel), dtype=torch.float32)

    return {
        "geo_scene_id": scene_id,
        "geo_change_ids": change_ids,
        "geo_nuisance_vec": nuis_vec,
        "geo_reliability": reliability,
    }

from torchvision import tv_tensors
from torchvision.transforms.v2 import InterpolationMode



class CDDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        resolution: int = 256,
        split: str = "train",
        data_len: int = -1,
        geopix_dirname: str = "json",  
        return_paths: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.split = split
        self.return_paths = return_paths

        list_path = self.root_dir / LIST_FOLDER_NAME / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"Missing list file: {list_path}")

        self.img_name_list = _load_img_name_list(list_path)
        self.dataset_len = len(self.img_name_list)

        if data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, data_len)

        
        self.geopix_map: Dict[str, Dict[str, Any]] = {}
        if geopix_dirname:
            self.geopix_map = _load_geopix_index(self.root_dir, geopix_dirname, split)
        
        if self.split == 'train':
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.GaussianBlur(kernel_size=3),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            ])



    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_name = self.img_name_list[index % self.data_len]
        a_path, b_path, l_path = _get_paths(self.root_dir, img_name)
        sample_id = _stem(img_name)

        img_A = Image.open(a_path).convert("RGB")
        img_B = Image.open(b_path).convert("RGB")
        img_label = Image.open(l_path).convert("RGB")

        
        img_A, img_B, img_label = self.transform(img_A, img_B, img_label)
         
        img_A, img_B = img_A * 2 - 1, img_B * 2 - 1
        img_label = (img_label > 0.5).long()
        if img_label.dim() > 2: img_label = img_label[0]


        geopix = self.geopix_map.get(sample_id, None)
        geo_t = _geopix_to_tensors(geopix)

        out: Dict[str, Any] = {
            "A": img_A,
            "B": img_B,
            "L": img_label,
            "Index": index,
            "id": sample_id,
            
            "geopix": geopix if geopix is not None else {},
            
            **geo_t,
        }

        if self.return_paths:
            out.update({"A_path": str(a_path), "B_path": str(b_path), "L_path": str(l_path)})

        return out

class SemanticCDDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        resolution: int = 256,
        split: str = "train",
        data_len: int = -1,
        geopix_dirname: str = "json",  
        return_paths: bool = False,
        ignore_index: int | None = None,  
    ):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.split = split
        self.return_paths = return_paths
        self.ignore_index = ignore_index

        list_path = self.root_dir / LIST_FOLDER_NAME / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"Missing list file: {list_path}")

        self.img_name_list = _load_img_name_list(list_path)
        self.dataset_len = len(self.img_name_list)

        if data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, data_len)

        
        self.geopix_map: Dict[str, Dict[str, Any]] = {}
        if geopix_dirname:
            self.geopix_map = _load_geopix_index(self.root_dir, geopix_dirname, split)

        
        
        
        
        
        if self.split == "train":
            self.geom = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                
            ])
            self.img_aug = v2.Compose([
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.GaussianBlur(kernel_size=3),
            ])
        else:
            self.geom = v2.Identity()
            self.img_aug = v2.Identity()

        
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_name = self.img_name_list[index % self.data_len]
        a_path, b_path, l_path = _get_paths(self.root_dir, img_name)
        sample_id = _stem(img_name)

        img_A = Image.open(a_path).convert("RGB")
        img_B = Image.open(b_path).convert("RGB")

        
        
        img_label_pil = Image.open(l_path)
        if img_label_pil.mode != "L":
            img_label_pil = img_label_pil.convert("L")

        
        img_A, img_B, img_label_pil = self.geom(img_A, img_B, img_label_pil)

        
        img_A, img_B = self.img_aug(img_A), self.img_aug(img_B)

        
        img_A = self.to_tensor(img_A) * 2 - 1
        img_B = self.to_tensor(img_B) * 2 - 1

        
        lab = np.array(img_label_pil, dtype=np.uint8)  
        img_label = torch.from_numpy(lab).long()       

        
        

        geopix = self.geopix_map.get(sample_id, None)
        geo_t = _geopix_to_tensors(geopix)

        out: Dict[str, Any] = {
            "A": img_A,
            "B": img_B,
            "L": img_label,
            "Index": index,
            "id": sample_id,
            "geopix": geopix if geopix is not None else {},
            **geo_t,
        }

        if self.return_paths:
            out.update({"A_path": str(a_path), "B_path": str(b_path), "L_path": str(l_path)})

        return out

class SCDDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        resolution: int = 512,
        split: str = "train",
        data_len: int = -1,
        geopix_dirname: str = "geopix_json_openai",
        return_paths: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.split = split
        self.return_paths = return_paths

        list_path = self.root_dir / LIST_FOLDER_NAME / f"{split}.txt"
        if not list_path.exists():
            raise FileNotFoundError(f"Missing list file: {list_path}")

        self.img_name_list = _load_img_name_list(list_path)
        self.dataset_len = len(self.img_name_list)
        if data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, data_len)

        self.geopix_map: Dict[str, Dict[str, Any]] = {}
        if geopix_dirname:
            self.geopix_map = _load_geopix_index(self.root_dir, geopix_dirname, split)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_name = self.img_name_list[index % self.data_len]
        a_path, b_path, l_path = _get_paths(self.root_dir, img_name)
        sample_id = _stem(img_name)

        img_A = Image.open(a_path).convert("RGB")
        img_B = Image.open(b_path).convert("RGB")

        
        l1_path = self.root_dir / "label1" / img_name
        l2_path = self.root_dir / "label2" / img_name

        img_label = np.array(Image.open(l_path), dtype=np.uint8)
        img_label1 = np.array(Image.open(l1_path), dtype=np.uint8) if l1_path.exists() else None
        img_label2 = np.array(Image.open(l2_path), dtype=np.uint8) if l2_path.exists() else None

        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))

        img_label = torch.from_numpy(img_label)
        if img_label.dim() > 2:
            img_label = img_label[0]

        out: Dict[str, Any] = {
            "A": img_A,
            "B": img_B,
            "L": img_label,
            "Index": index,
            "id": sample_id,
        }

        
        if img_label1 is not None:
            img_label1 = torch.from_numpy(img_label1)
            if img_label1.dim() > 2:
                img_label1 = img_label1[0]
            cls_category1 = torch.unique(img_label1)
            cls_label1 = torch.zeros(7, dtype=torch.int64)
            for v in cls_category1:
                vv = int(v.item())
                if 0 <= vv < 7:
                    cls_label1[vv] = 1
            out.update({"L1": img_label1, "cls1": cls_label1})

        if img_label2 is not None:
            img_label2 = torch.from_numpy(img_label2)
            if img_label2.dim() > 2:
                img_label2 = img_label2[0]
            cls_category2 = torch.unique(img_label2)
            cls_label2 = torch.zeros(7, dtype=torch.int64)
            for v in cls_category2:
                vv = int(v.item())
                if 0 <= vv < 7:
                    cls_label2[vv] = 1
            out.update({"L2": img_label2, "cls2": cls_label2})

        geopix = self.geopix_map.get(sample_id, None)
        geo_t = _geopix_to_tensors(geopix)
        out.update({"geopix": geopix if geopix is not None else {}, **geo_t})

        if self.return_paths:
            out.update({"A_path": str(a_path), "B_path": str(b_path), "L_path": str(l_path)})

        return out


if __name__ == "__main__":
    
    root_dir = "/home/zhanghaoyu/datasets/LEVIRCD/LEVIR-CD256"
    ds = CDDataset(root_dir=root_dir, split="val", geopix_dirname="geopix_json_openai", return_paths=True)
    x = ds[0]
    print("keys:", list(x.keys()))
    print("id:", x["id"])
    print("geo_scene_id:", x["geo_scene_id"].item(), "geo_change_ids:", x["geo_change_ids"].tolist())
    print("geo_nuisance_vec:", x["geo_nuisance_vec"].tolist(), "geo_reliability:", float(x["geo_reliability"]))
