import os
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.utils.data._utils.collate import default_collate

import data as Data
import models as Model
import core.logger as Logger


def cd_collate_fn(batch):
    for b in batch:
        b.pop("geopix", None)
    return default_collate(batch)


SCENE_TYPES = ["urban", "suburban", "rural", "forest", "farmland", "water", "mixed", "unknown"]
NUISANCE_TYPES = ["shadow", "illumination", "season", "misalignment", "cloud", "sensor_noise"]


def build_prompts_from_batch(batch):
    scene_ids = batch["geo_scene_id"].tolist()
    nuis_vec = batch["geo_nuisance_vec"].tolist()

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


def update_confusion_matrix(cm: np.ndarray, pred: np.ndarray, label: np.ndarray, num_classes: int):
    """
    pred:  (N,H,W) or (H,W)
    label: (N,H,W) or (H,W)
    rows = GT, cols = Pred
    """
    pf = pred.reshape(-1)
    lf = label.reshape(-1)

    
    valid = (lf >= 0) & (lf < num_classes) & (pf >= 0) & (pf < num_classes)

    
    binc = np.bincount(
        num_classes * lf[valid].astype(np.int64) + pf[valid].astype(np.int64),
        minlength=num_classes * num_classes
    )
    cm += binc.reshape(num_classes, num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--phase", type=str, default="val", choices=["val", "test", "infer"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="cm_slcd_ours.npy")
    parser.add_argument("--gpu_ids", type=str, default=None)  
    args = parser.parse_args()

    rank, world, device, is_ddp = ddp_init(
        int(os.environ.get("LOCAL_RANK", args.local_rank))
    )

    opt = Logger.dict_to_nonedict(Logger.parse(args))

    logger = logging.getLogger("base")
    if rank == 0:
        Logger.setup_logger(None, opt["path_cd"]["log"], "cm_only", logging.INFO, True)

    
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

    num_classes = int(opt["model"]["n_classes"])
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            img1 = batch["A"].to(device, non_blocking=True)
            img2 = batch["B"].to(device, non_blocking=True)
            gt = batch["L"].to(device, non_blocking=True).long()

            
            if num_classes == 2:
                gt = gt.clone()
                gt[gt > 0] = 1

            prompts = build_prompts_from_batch(batch)

            with autocast(dtype=torch.bfloat16):
                pred, _ = model(img1, img2, prompts)

            mask = torch.argmax(pred, dim=1)

            update_confusion_matrix(
                cm,
                mask.detach().cpu().numpy(),
                gt.detach().cpu().numpy(),
                num_classes=num_classes,
            )

            if rank == 0 and step % 20 == 0:
                logger.info(f"[{args.phase}] step {step}/{len(loader)}")

    
    if is_ddp:
        cm_tensor = torch.from_numpy(cm).to(device)
        dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
        cm = cm_tensor.cpu().numpy()

    if rank == 0:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        np.save(args.save_path, cm)
        logger.info(f"Confusion matrix saved to: {args.save_path}")
        logger.info(f"cm shape: {cm.shape}")
        logger.info(f"\n{cm}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()