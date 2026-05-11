import os
import torch

def load_remoteclip(
    model_name: str = "ViT-L-14",   
    device: str | torch.device = "cuda",
    hf_repo: str = "chendelong/RemoteCLIP",
    cache_dir: str = "checkpoints",
):
    """
    按 RemoteCLIP 官方 README：OpenCLIP create_model_and_transforms + 加载 RemoteCLIP-*.pt 权重
    依赖:
      pip install open-clip-torch huggingface_hub
    """
    import open_clip
    from huggingface_hub import hf_hub_download

    device = torch.device(device) if not isinstance(device, torch.device) else device

    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    
    os.makedirs(cache_dir, exist_ok=True)
    ckpt_name = f"RemoteCLIP-{model_name}.pt"
    ckpt_path = hf_hub_download(hf_repo, ckpt_name, cache_dir=cache_dir)

    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=True)

    
    model = model.to(device).eval()

    return model, preprocess, tokenizer, ckpt_path, msg
