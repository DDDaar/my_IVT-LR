from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


_PRESET_TO_HF = {
    ("sam", "vit_h"): "facebook/sam-vit-huge",
    ("sam", "vit_l"): "facebook/sam-vit-large",
    ("sam", "vit_b"): "facebook/sam-vit-base",
    
    ("dinov2", "dinov2-large"): "facebook/dinov2-large", # 对应 ViT-L
    ("dinov2", "dinov2-base"): "facebook/dinov2-base",  # 对应 ViT-B
    ("dinov2", "dinov2-small"): "facebook/dinov2-small", # 对应 ViT-S

    # DepthAnything v2 vitl: repo naming varies; override with explicit `hf:` if needed.
    ("depth", "depth_anything_v2_vitl"): "depth-anything/Depth-Anything-V2-Large",
}


def resolve_teacher_source(expert: str, source: str) -> str:
    """
    Allows short preset names like:
    - sam: "vit_h"
    - dinov2: "dinov2_vitl14"
    - depth: "depth_anything_v2_vitl"
    """
    if not source:
        raise ValueError("Empty model source")
    key = (expert, source.strip().lower())
    if key in _PRESET_TO_HF:
        return "hf:" + _PRESET_TO_HF[key]
    return source


def resolve_model_source(source: str) -> str:
    """
    Resolve a model source into a local directory.

    Supported forms:
    - "local:/abs/path"
    - "hf:repo_id" (downloads via huggingface_hub)
    - "/abs/path" (treated as local)
    """
    if not source:
        raise ValueError("Empty model source")

    if source.startswith("local:"):
        path = source[len("local:") :]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local model path not found: {path}")
        return path

    if source.startswith("hf:"):
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is not available; cannot download hf: models")
        repo_id = source[len("hf:") :]
        return snapshot_download(repo_id=repo_id)

    # Fallback: treat as local path.
    if not os.path.exists(source):
        raise FileNotFoundError(f"Model path not found: {source}")
    return source


class OnlineTeacherManager:
    """
    Online teacher extraction (best-effort).

    Notes:
    - This implementation targets models available via HuggingFace Transformers.
    - For SAM, a native Segment-Anything implementation is usually required; we currently do not
      implement SAM online extraction here. Use offline cache for SAM, or extend this class.
    - Input uses `pixel_values` (already derived from the image). If you want raw-image teacher
      preprocessing, extend the dataloader to provide images/paths and add expert-specific processors.
    """

    def __init__(self, teacher_sources: Dict[str, Optional[str]], device: str = "cuda", dtype: str = "bf16"):
        self.teacher_sources = dict(teacher_sources)
        self.device = device
        self.dtype = dtype
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}

    def _load_model(self, expert: str):
        if expert in self._models:
            return self._models[expert]

        src = self.teacher_sources.get(expert)
        if not src:
            raise ValueError(f"Missing online teacher source for expert={expert}")

        src = resolve_teacher_source(expert, src)
        path = resolve_model_source(src)

        if expert == "sam":
            # Requires raw images + SamProcessor preprocessing.
            from transformers import SamModel, SamProcessor

            processor = SamProcessor.from_pretrained(path)
            model = SamModel.from_pretrained(path)
            self._processors[expert] = processor
        else:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(path, trust_remote_code=True)

        model.eval()
        model.to(self.device)
        if self.dtype in ("bf16", "bfloat16"):
            model.to(torch.bfloat16)
        elif self.dtype in ("fp16", "float16"):
            model.to(torch.float16)
        else:
            model.to(torch.float32)

        self._models[expert] = model
        return model

    @torch.no_grad()
    def get_vector(self, expert: str, *, pixel_values: Optional[torch.Tensor] = None, images: Any = None) -> torch.Tensor:
        """
        Returns a per-image vector: shape (B, d).
        """
        model = self._load_model(expert)

        # ====== [新增] 获取当前模型实际的 dtype ======
        model_dtype = next(model.parameters()).dtype
        
        if expert == "sam":
            if images is None:
                raise ValueError("SAM online teacher requires raw images (PIL/np arrays) passed as `images`.")
            processor = self._processors.get("sam")
            if processor is None:
                raise RuntimeError("SAM processor is not loaded")

            inputs = processor(images=images, return_tensors="pt")
            
            # ====== [修改] 强制将输入对齐到 model_dtype ======
            pv = inputs["pixel_values"].to(self.device).to(model_dtype)

            # ====== [修改核心] 使用专用的提取方法，直接获取特征并跳过解码器 ======
            img_emb = model.get_image_embeddings(pv)
            
            if img_emb is None:
                raise RuntimeError("Failed to extract image embeddings using get_image_embeddings().")
                
            vec = img_emb.mean(dim=(2, 3))  # (B, 256)
            return vec.detach()

        if pixel_values is None:
            raise ValueError(f"expert={expert} requires pixel_values for online extraction.")

        # pixel_values can be (B, 1, 3, H, W) for single-image setups. Flatten that.
        if pixel_values.ndim == 5 and pixel_values.size(1) == 1:
            pv = pixel_values[:, 0]
        else:
            pv = pixel_values

        # ====== [修改] 同样强制将输入对齐到 model_dtype ======
        pv = pv.to(self.device).to(model_dtype)

        out = model(pixel_values=pv)

        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            hss = getattr(out, "hidden_states", None)
            if hss is not None and len(hss) > 0:
                hs = hss[-1]

        if hs is None:
            raise RuntimeError(f"Cannot extract hidden states for expert={expert} from output type={type(out)}")

        if hs.ndim == 3:
            vec = hs.mean(dim=1)
        elif hs.ndim == 2:
            vec = hs
        else:
            vec = hs.view(hs.size(0), -1)

        return vec.detach()


def infer_teacher_dim_from_source(expert: str, source: str) -> int:
    """
    Infer teacher vector dimension without requiring YAML `teacher_dims`.

    - sam: returns 256 for `image_embeddings` pooled vector
    - dinov2/depth: tries AutoConfig.hidden_size and common fallbacks
    """
    src = resolve_teacher_source(expert, source)
    path = resolve_model_source(src)

    if expert == "sam":
        # SamModel image_embeddings channel dim is 256 for vit-b/l/h in transformers.
        return 256

    
    # ====== 新增对 depth 的维度硬编码拦截 ======
    if expert == "depth":
        path_lower = path.lower()
        src_lower = src.lower()
        if "large" in path_lower or "vitl" in src_lower:
            return 1024
        elif "base" in path_lower or "vitb" in src_lower:
            return 768
        elif "small" in path_lower or "vits" in src_lower:
            return 384
        return 1024  # 默认 fallback 到 1024
    # ============================================
    
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    for attr in ("hidden_size", "vision_hidden_size", "embed_dim", "dim"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v

    # Common nested configs
    for nested in ("vision_config", "encoder_config", "model_config"):
        nc = getattr(cfg, nested, None)
        if nc is None:
            continue
        v = getattr(nc, "hidden_size", None)
        if isinstance(v, int) and v > 0:
            return v

    raise RuntimeError(
        f"Cannot infer teacher dim for expert={expert} from config. "
        f"Please set `expert.teacher_dims.{expert}` explicitly or use a different model source."
    )
