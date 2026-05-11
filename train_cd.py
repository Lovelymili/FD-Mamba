import os
import torch
import torch.distributed as dist
import argparse
import logging
import core.logger as Logger

import numpy as np
import data as Data
import models as Model
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import ce_dice, cross_entropy, dice, ce2_dice1, ce1_dice2
from misc.torchutils import get_scheduler, save_network
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import autocast

from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            
            curr_epoch = self.last_epoch - self.warmup_epochs
            total_epochs = self.max_epochs - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * curr_epoch / total_epochs)) / 2
                    for base_lr in self.base_lrs]

from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F
def cd_collate_fn(batch):
    """
    Fix default_collate crash caused by variable-length dict/list fields (e.g., raw 'geopix').
    Keep everything collatable, but drop or keep 'geopix' as a list.
    """
    
    for b in batch:
        if "geopix" in b:
            b.pop("geopix", None)

    return default_collate(batch)

    
    
    
    
    
    
    



def main():
    
    SCENE_TYPES = ["urban", "suburban", "rural", "forest", "farmland", "water", "mixed", "unknown"]

    NUISANCE_TYPES = ["shadow", "illumination", "season", "misalignment", "cloud", "sensor_noise"]

    def build_prompts_from_batch(train_data) -> list[str]:
        """
        修改后的 Prompt 构建函数：
        1. 移除 CHANGE 字段，杜绝标签泄露。
        2. 保留 SCENE (上下文) 和 NUISANCE (干扰项)。
        3. 逻辑转变：由 "寻找变化" 变为 "在特定场景下忽略特定干扰"。
        """
        scene_ids = train_data["geo_scene_id"].tolist()          
        nuis_vec = train_data["geo_nuisance_vec"].tolist()       

        prompts = []
        for s, nv in zip(scene_ids, nuis_vec):
            s = int(s)
            
            scene = SCENE_TYPES[s] if 0 <= s < len(SCENE_TYPES) else "unknown"

            
            nuisances = []
            for i, v in enumerate(nv):
                if float(v) > 0.5:  
                    nuisances.append(NUISANCE_TYPES[i])
        
            
            if nuisances:
                instruction = f"Ignore {', '.join(nuisances)}."
            else:
                instruction = "Clear conditions."

            
            prompts.append(f"Satellite image of {scene} area. {instruction}")
        
        return prompts

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/levir+_cdmamba/levir+_cdmamba.json')
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    rank = int(os.environ.get('RANK', 0))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    
    if rank == 0:
        Logger.setup_logger(
            logger_name=None,
            root=opt['path_cd']['log'],
            phase='train',
            level=logging.INFO,
            screen=True
        )
        logger = logging.getLogger('base')
    else:
        logger = logging.getLogger('base')
        logger.setLevel(logging.ERROR)

    
    dataloaders = {}
    print('Building dataloaders...')
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase in ['train', 'val']:
            logger.info(f"Creating [{phase}] change-detection dataset and dataloader.")
            if opt['model']['n_classes'] > 2:
                logger.info(f"Creating [{phase}] change-detection dataset and dataloader with multi-class labels.")
                dataset = Data.create_cd_dataset(dataset_opt=dataset_opt, phase=phase, type='semantic')
            else:
                logger.info(f"Creating [{phase}] change-detection dataset and dataloader with binary labels.")
                dataset = Data.create_cd_dataset(dataset_opt=dataset_opt, phase=phase, type='binary')
            sampler = DistributedSampler(dataset, shuffle=(phase == 'train')) if local_rank != -1 else None
            if sampler is not None:
                dataset_opt['use_shuffle'] = False
            dataloaders[phase] = Data.create_cd_dataloader(dataset, dataset_opt, phase, sampler=sampler, collate_fn=cd_collate_fn)

    
    cd_model = Model.create_CD_model(opt).to(device)
    if local_rank != -1:
        cd_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cd_model)

    if local_rank != -1:
        dist.barrier()
        cd_model = DDP(cd_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    
    metric = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])

    
    if opt['phase'] == 'train':
        best_F1 = 0.0
        loss_map = {
            'ce_dice': ce_dice,
            'ce': cross_entropy,
            'dice': dice,
            'ce2_dice1': ce2_dice1,
            'ce1_dice2': ce1_dice2
        }
        loss_fun = loss_map.get(opt['model']['loss'], ce_dice)

        
        model_params = cd_model.parameters() if not hasattr(cd_model, "module") else cd_model.module.parameters()
        params = [p for p in model_params if p.requires_grad]
        optimer = torch.optim.AdamW(params, lr=opt['train']["optimizer"]["lr"])

        
        scheduler = WarmupCosineAnnealingLR(
            optimer,
            warmup_epochs=opt['train'].get('warmup_epochs', 10),
            max_epochs=opt['train']['n_epoch']
        )

        for current_epoch in range(opt['train']['n_epoch']):
            
            if local_rank != -1 and hasattr(dataloaders['train'], "sampler") and dataloaders['train'].sampler is not None:
                dataloaders['train'].sampler.set_epoch(current_epoch)

            cd_model.train()
            metric.clear()

            
            curr_lr = optimer.param_groups[0]['lr']
            if rank == 0:
                logger.info(f"--- Epoch {current_epoch} Start | LR: {curr_lr:.6f} ---")

            for current_step, train_data in enumerate(dataloaders['train']):
                img1 = train_data['A'].to(device, non_blocking=True)
                img2 = train_data['B'].to(device, non_blocking=True)
                gt = train_data['L'].to(device, non_blocking=True).long()
                if opt['model']['n_classes'] == 2:
                    gt = (gt > 0).long()

                
                geo_reliability = train_data["geo_reliability"].to(device, non_blocking=True).view(-1, 1)
                geo_nuis = train_data["geo_nuisance_vec"].to(device, non_blocking=True)
                meta = torch.cat([geo_reliability, geo_nuis], dim=1)

                
                prompts = build_prompts_from_batch(train_data)

                optimer.zero_grad(set_to_none=True)
                with autocast(dtype=torch.bfloat16, enabled=True):
                    
                    pred, score_map = cd_model(img1, img2, meta=meta, prompts=prompts)
                    

                    
                    main_loss = loss_fun(pred.float(), gt)
                    

                    
                    with torch.no_grad():
                        prob = torch.softmax(pred.float(), dim=1)                 
                        conf = prob.max(dim=1, keepdim=True).values              
                        tau = 0.80                                                
                        hard = (conf < tau).float()                               

                    
                    aux_ce_pix = F.cross_entropy(score_map.float(), gt.squeeze(1).long(), reduction="none")  
                    aux_loss = (aux_ce_pix * hard.squeeze(1)).sum() / (hard.sum() + 1e-6)
    
                    
                    loss = main_loss + 0.4 * aux_loss

                    
                    
                   

                loss.backward()
                torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=1.0)
                optimer.step()

                
                with torch.no_grad():
                    G_pred = torch.argmax(pred.detach(), dim=1)
                    metric.update_cm(pr=G_pred.cpu().numpy(), gt=gt.detach().cpu().numpy())

                    if rank == 0 and current_step % 50 == 0:
                        unique_pred = torch.unique(G_pred)
                        if unique_pred.numel() < 2:
                            logger.info(f"Step {current_step}: Pred is all background (only {unique_pred.tolist()})")

                if rank == 0 and current_step % opt['train']['train_print_iter'] == 0:
                    logger.info(f'Epoch: {current_epoch} [{current_step}/{len(dataloaders["train"])}] Main_Loss: {main_loss.item():.5f}  Loss: {loss.item():.5f}')

            
            if local_rank != -1:
                train_cm_tensor = torch.as_tensor(metric.sum, device=device)
                dist.all_reduce(train_cm_tensor, op=dist.ReduceOp.SUM)
                metric.sum = train_cm_tensor.detach().cpu().numpy()

            if rank == 0:
                train_scores = metric.get_scores()
                logger.info(
                    f'[Train Epoch Summary] Epoch: {current_epoch}, '
                    f'Change-F1: {train_scores["F1_1"]:.5f}, Change-IoU: {train_scores["iou_1"]:.5f}'
                )

            
            
            
            if current_epoch % opt['train']['val_freq'] == 0:
                cd_model.eval()
                metric.clear()

                with torch.no_grad():
                    for val_data in dataloaders['val']:
                        v1 = val_data['A'].to(device, non_blocking=True)
                        v2 = val_data['B'].to(device, non_blocking=True)
                        vgt = val_data['L'].to(device, non_blocking=True).long()
                        if opt['model']['n_classes'] == 2:
                            vgt = (vgt > 0).long()

                        
                        v_rel = val_data["geo_reliability"].to(device, non_blocking=True).view(-1, 1)
                        v_nuis = val_data["geo_nuisance_vec"].to(device, non_blocking=True)
                        v_meta = torch.cat([v_rel, v_nuis], dim=1)

                        
                        v_prompts = build_prompts_from_batch(val_data)

                        with autocast(dtype=torch.bfloat16, enabled=True):
                            vpred, _ = cd_model(v1, v2, meta=v_meta, prompts=v_prompts)
                            
                        v_G_pred = torch.argmax(vpred.detach(), dim=1)
                        metric.update_cm(pr=v_G_pred.cpu().numpy(), gt=vgt.cpu().numpy())

                if local_rank != -1:
                    val_cm_tensor = torch.as_tensor(metric.sum, device=device)
                    dist.all_reduce(val_cm_tensor, op=dist.ReduceOp.SUM)
                    metric.sum = val_cm_tensor.detach().cpu().numpy()

                if rank == 0:
                    val_scores = metric.get_scores()
                    val_F1 = val_scores['F1_1']
                    val_IoU = val_scores['iou_1']
                    logger.info(
                        f'[Validation Summary] Epoch: {current_epoch}, '
                        f'Change-F1: {val_F1:.5f}, Change-IoU: {val_IoU:.5f}'
                    )

                    is_best = val_F1 > best_F1
                    if is_best:
                        best_F1 = val_F1
                        logger.info(f'---> Best model updated at Epoch {current_epoch}, F1: {best_F1:.5f}')

                    model_to_save = cd_model.module if hasattr(cd_model, 'module') else cd_model
                    save_network(opt, current_epoch, model_to_save, optimer, is_best)

            scheduler.step()

    
    if local_rank != -1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()