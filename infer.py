import os
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast

import data as Data
import models as Model
import core.logger as Logger

from misc.metric_tools import ConfuseMatrixMeter
from torch.utils.data._utils.collate import default_collate


def cd_collate_fn(batch):
    for b in batch:
        b.pop("geopix", None)
    return default_collate(batch)



SCENE_TYPES = ["urban", "suburban", "rural", "forest", "farmland", "water", "mixed", "unknown"]
NUISANCE_TYPES = ["shadow", "illumination", "season", "misalignment", "cloud", "sensor_noise"]

def build_prompts_from_batch(batch):
    scene_ids = batch["geo_scene_id"].tolist()
    nuis_vec  = batch["geo_nuisance_vec"].tolist()

    prompts = []
    for s, nv in zip(scene_ids, nuis_vec):
        scene = SCENE_TYPES[int(s)] if 0 <= int(s) < len(SCENE_TYPES) else "unknown"
        nuis = [NUISANCE_TYPES[i] for i, v in enumerate(nv) if float(v) > 0.5]
        if nuis:
            prompts.append(f"Satellite image of {scene} area. Ignore {', '.join(nuis)}.")
        else:
            prompts.append(f"Satellite image of {scene} area.")
    return prompts



def load_cd_checkpoint(model, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "net", "cd_model"]:
            if k in ckpt:
                ckpt = ckpt[k]
                break

    if list(ckpt.keys())[0].startswith("module."):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    return model.load_state_dict(ckpt, strict=strict)


def ddp_init(local_rank):
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        dist.barrier()
        return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank), True
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"), False

def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)

def collapse_cm_to_binary(cm: np.ndarray) -> np.ndarray:
    """
    Collapse multi-class CM (KxK) into binary:
      class 0 = background/unchanged
      class 1 = foreground/changed (classes 1..K-1)
    Convention: rows=GT, cols=Pred
    Return 2x2: [[TN, FP],
                 [FN, TP]]
    """
    cm = cm.astype(np.float64)
    K = cm.shape[0]
    assert cm.shape[0] == cm.shape[1], f"CM must be square, got {cm.shape}"

    tn = cm[0, 0]
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    tp = cm[1:, 1:].sum()
    return np.array([[tn, fp],
                     [fn, tp]], dtype=np.float64)
def _summarize_binary_from_cm(cm: np.ndarray) -> dict:
    """
    Summarize metrics from a 2x2 confusion matrix.

    Expected format (rows=GT, cols=Pred):

        [[TN, FP],
         [FN, TP]]

    Parameters
    ----------
    cm : np.ndarray
        Shape must be (2,2)

    Returns
    -------
    dict with:
        TP, FP, FN, TN
        OA (overall accuracy)
        P (precision for positive class)
        R (recall / TPR)
        Spec (specificity / TNR)
        F1
        BalAcc
        Kappa
        PredPosRatio
        GtPosRatio
    """

    if cm.shape != (2, 2):
        raise ValueError(
            f"_summarize_binary_from_cm expects (2,2) matrix, "
            f"but got shape {cm.shape}"
        )

    cm = cm.astype(np.float64)

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    total = cm.sum()

    
    oa = _safe_div(tp + tn, total)

    
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)        
    specificity = _safe_div(tn, tn + fp)   

    
    f1 = _safe_div(2 * precision * recall, precision + recall)

    
    bal_acc = 0.5 * (recall + specificity)

    
    pe = _safe_div(
        (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn),
        total * total
    )
    kappa = _safe_div(oa - pe, 1.0 - pe)

    
    pred_pos = tp + fp
    gt_pos = tp + fn

    pred_pos_ratio = _safe_div(pred_pos, total)
    gt_pos_ratio = _safe_div(gt_pos, total)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "OA": oa,
        "P": precision,
        "R": recall,
        "Spec": specificity,
        "F1": f1,
        "BalAcc": bal_acc,
        "Kappa": kappa,
        "PredPosRatio": pred_pos_ratio,
        "GtPosRatio": gt_pos_ratio,
    }
def _log_metrics_multiclass_and_cd(logger, tag: str, metric: ConfuseMatrixMeter):
    s = metric.get_scores()
    cm = metric.sum  

    K = cm.shape[0]
    logger.info(f"===== {tag} (MULTI-CLASS K={K}) =====")
    logger.info(f"acc={float(s.get('acc', 0.0)):.5f}  miou={float(s.get('miou', 0.0)):.5f}  mf1={float(s.get('mf1', 0.0)):.5f}")

    
    for k in range(K):
        iouk = float(s.get(f"iou_{k}", 0.0))
        f1k  = float(s.get(f"F1_{k}", 0.0))
        pk   = float(s.get(f"precision_{k}", 0.0))
        rk   = float(s.get(f"recall_{k}", 0.0))
        logger.info(f"class{k}: IoU={iouk:.5f}  F1={f1k:.5f}  P={pk:.5f}  R={rk:.5f}")

    
    if "SCD_Sek" in s or "Fscd" in s or "SCD_IoU_mean" in s:
        logger.info(
            f"SCD: Sek={float(s.get('SCD_Sek', 0.0)):.5f}  "
            f"Fscd={float(s.get('Fscd', 0.0)):.5f}  "
            f"SCD_IoU_mean={float(s.get('SCD_IoU_mean', 0.0)):.5f}"
        )

    
    c2 = collapse_cm_to_binary(cm)
    binm = _summarize_binary_from_cm(c2)
    logger.info(f"===== {tag} (COLLAPSED CD: 0 vs non-0) =====")
    logger.info(
        "CM[[TN,FP],[FN,TP]]="
        f"[[{int(binm['TN'])},{int(binm['FP'])}],[{int(binm['FN'])},{int(binm['TP'])}]]"
    )
    logger.info(
        f"OA={binm['OA']:.5f}  Kappa={binm['Kappa']:.5f}  BalAcc={binm['BalAcc']:.5f}  "
        f"PredChange%={100*binm['PredPosRatio']:.2f}  GtChange%={100*binm['GtPosRatio']:.2f}"
    )
    logger.info(
        f"Change(P/R/F1)={binm['P']:.5f}/{binm['R']:.5f}/{binm['F1']:.5f}  Spec={binm['Spec']:.5f}"
    )

def _make_palette(n_classes: int):
    """
    Return palette as uint8 array [n_classes, 3] in RGB.
    Class 0 is black by default.
    """
    palette = np.zeros((n_classes, 3), dtype=np.uint8)

    
    base = np.array([
        [0, 0, 0],        
        [255, 0, 0],      
        [0, 255, 0],      
        [0, 0, 255],      
        [255, 255, 0],    
        [255, 0, 255],    
        [0, 255, 255],    
        [255, 128, 0],    
        [128, 0, 255],    
        [0, 128, 255],    
        [128, 255, 0],    
        [255, 0, 128],    
    ], dtype=np.uint8)

    m = min(n_classes, len(base))
    palette[:m] = base[:m]

    
    
    for cid in range(len(base), n_classes):
        r = g = b = 0
        x = cid
        for i in range(8):
            r |= ((x >> 0) & 1) << (7 - i)
            g |= ((x >> 1) & 1) << (7 - i)
            b |= ((x >> 2) & 1) << (7 - i)
            x >>= 3
        palette[cid] = np.array([r, g, b], dtype=np.uint8)

    return palette

def colorize_mask(mask_hw: np.ndarray, n_classes: int) -> np.ndarray:
    """
    mask_hw: (H, W) int
    return: (H, W, 3) uint8 in BGR (for cv2.imwrite)
    """
    mask_hw = mask_hw.astype(np.int64)
    palette = _make_palette(n_classes)  
    mask_hw = np.clip(mask_hw, 0, n_classes - 1)

    rgb = palette[mask_hw]  
    bgr = rgb[..., ::-1].copy()  
    return bgr

def main():
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--phase", type=str, default="val", choices=["val", "test", "infer"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--save_dir", type=str, default="pred_masks")
    parser.add_argument("--gpu_ids", type=str, default=None)  

    args = parser.parse_args()

    rank, world, device, is_ddp = ddp_init(
        int(os.environ.get("LOCAL_RANK", args.local_rank))
    )

    opt = Logger.dict_to_nonedict(Logger.parse(args))
    logger = logging.getLogger("base")
    if rank == 0:
        Logger.setup_logger(None, opt["path_cd"]["log"], "infer", logging.INFO, True)

    
    dataset_opt = opt["datasets"].get(args.phase, opt["datasets"]["val"])
    dataset = Data.create_cd_dataset(dataset_opt, args.phase)
    sampler = DistributedSampler(dataset, shuffle=False) if is_ddp else None
    loader = Data.create_cd_dataloader(
        dataset, dataset_opt, args.phase, sampler=sampler, collate_fn=cd_collate_fn
    )

    
    model = Model.create_CD_model(opt).to(device).eval()

    load_cd_checkpoint(model, args.checkpoint)

    if is_ddp:
        model = DDP(model, device_ids=[device.index])

    metric = ConfuseMatrixMeter(n_class=opt["model"]["n_classes"])

    with torch.no_grad():
        for step, batch in enumerate(loader):
            img1 = batch["A"].to(device)
            img2 = batch["B"].to(device)
            gt   = batch["L"].to(device).long()
            if opt['model']['n_classes'] == 2:
                gt[gt > 0] = 1
            meta = torch.cat(
                [
                    batch["geo_reliability"].to(device).view(-1, 1),
                    batch["geo_nuisance_vec"].to(device),
                ],
                dim=1,
            )
            prompts = build_prompts_from_batch(batch)

            with autocast(dtype=torch.bfloat16):
                pred,_ = model(img1, img2, prompts)

            mask = torch.argmax(pred, dim=1)
            metric.update_cm(mask.cpu().numpy(), gt.cpu().numpy())

            if args.save_mask and rank == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                n_classes = int(opt["model"]["n_classes"])
                for i, name in enumerate(batch.get("id", [])):
                    mi = mask[i].cpu().numpy()

                    if n_classes > 2:
                        vis = colorize_mask(mi, n_classes)          
                    else:
                        vis = (mi * 255).astype(np.uint8)           
                    cv2.imwrite(os.path.join(args.save_dir, f"{name}.png"), vis)
            gt_raw = batch["L"].to(device).long()
            if rank== 0 and step ==0:
                gt_raw0 = gt_raw[0].detach().cpu().numpy()
                logger.info(f"gt_raw[0] unique(before binarize) = {np.unique(gt_raw0)}")

            if rank == 0 and step % 20 == 0:
                logger.info(f"[{args.phase}] step {step}/{len(loader)}")

    if is_ddp:
        cm = torch.from_numpy(metric.sum).to(device)
        dist.all_reduce(cm)
        metric.sum = cm.cpu().numpy()

    if rank == 0:
        s = metric.get_scores()
        logger.info("===== CD RESULT =====")
        logger.info(f"F1_change={s['F1_1']:.5f}  IoU_change={s['iou_1']:.5f}")
        _log_metrics_multiclass_and_cd(logger, "CD", metric)


    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
