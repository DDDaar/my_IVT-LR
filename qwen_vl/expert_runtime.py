from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_EXPERTS = ("sam", "dinov2", "depth")


def _get_nested(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    return bool(v)


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return default
    return int(v)


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return default
    return float(v)


def _ensure_single_token(tokenizer, token_str: str) -> int:
    # We need token_str to map to exactly one token id, otherwise CE/align positions won't match.
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expert token must be a single token. token_str={token_str!r}, encoded_ids={ids}"
        )
    return ids[0]


@dataclass(frozen=True)
class ExpertRuntime:
    enabled: bool
    experts: Tuple[str, ...]
    token_ids: Dict[str, int]
    token_strs: Dict[str, str]
    slots: Dict[str, int]
    prefix_ids: Dict[str, List[int]]
    newline_ids: List[int]
    teacher_mode: str
    teacher_cache_dir: Optional[str]
    teacher_require: bool
    teacher_write_back: bool
    teacher_sources: Dict[str, Optional[str]]
    teacher_dims: Dict[str, int]
    align_enabled: bool
    align_weight: float
    align_loss: str
    normalize_teacher: bool
    normalize_student: bool

    def token_id_list(self) -> List[int]:
        return [self.token_ids[e] for e in self.experts]


def build_expert_runtime_from_config(
    config_dict: Dict[str, Any],
    tokenizer,
    *,
    add_tokens: bool,
    for_inference: bool,
) -> Optional[ExpertRuntime]:
    """
    Build a minimal, picklable expert runtime config.

    - `add_tokens=True` will add expert tokens to tokenizer (as normal tokens).
    - `for_inference=True` disables align by default but still constructs modules if dims provided.
    """
    expert_cfg = config_dict.get("expert") if isinstance(config_dict, dict) else None
    if not isinstance(expert_cfg, dict):
        return None

    enabled = _as_bool(expert_cfg.get("enabled"), False)
    if not enabled:
        return None

    experts = expert_cfg.get("experts", [])
    if not isinstance(experts, list) or len(experts) == 0:
        raise ValueError("expert.enabled=true but expert.experts is empty")

    experts_norm: List[str] = []
    for e in experts:
        if not isinstance(e, str):
            continue
        e2 = e.strip().lower()
        if e2 not in SUPPORTED_EXPERTS:
            raise ValueError(f"Unsupported expert name: {e!r}. Supported: {SUPPORTED_EXPERTS}")
        if e2 not in experts_norm:
            experts_norm.append(e2)

    token_strs: Dict[str, str] = dict(_get_nested(expert_cfg, ["token_strings"], {}) or {})
    prefix_text: Dict[str, str] = dict(_get_nested(expert_cfg, ["prefix_text"], {}) or {})
    slots: Dict[str, int] = dict(_get_nested(expert_cfg, ["slots"], {}) or {})

    # Optional: teacher_dims can be inferred from teacher model (train) or checkpoint (infer).
    teacher_dims: Dict[str, int] = dict(_get_nested(expert_cfg, ["teacher_dims"], {}) or {})

    # Teacher mode and cache.
    teacher_mode = str(_get_nested(expert_cfg, ["teacher", "mode"], "offline")).strip().lower()
    teacher_cache_dir = _get_nested(expert_cfg, ["teacher", "offline", "cache_dir"], None)
    if teacher_cache_dir is None:
        teacher_cache_dir = _get_nested(expert_cfg, ["teacher", "cache_dir"], None)
    teacher_require = _as_bool(_get_nested(expert_cfg, ["teacher", "hybrid", "require_teacher"], False), False)
    teacher_write_back = _as_bool(_get_nested(expert_cfg, ["teacher", "hybrid", "write_back"], False), False)
    teacher_sources: Dict[str, Optional[str]] = dict(_get_nested(expert_cfg, ["teacher", "online", "models"], {}) or {})

    # Align
    align_cfg = expert_cfg.get("align", {}) if isinstance(expert_cfg.get("align"), dict) else {}
    align_enabled = _as_bool(align_cfg.get("enabled"), True)
    align_weight = _as_float(align_cfg.get("weight"), 0.0)
    align_loss = str(align_cfg.get("loss", "mse")).strip().lower()
    normalize_teacher = _as_bool(align_cfg.get("normalize_teacher"), True)
    normalize_student = _as_bool(align_cfg.get("normalize_student"), True)

    if for_inference:
        # In inference we don't have teacher targets; we should not compute align loss.
        align_weight = 0.0

    # Add tokens as normal tokens so decode(skip_special_tokens=True) will keep them.
    if add_tokens:
        to_add = []
        for e in experts_norm:
            token_str = token_strs.get(e)
            if not token_str:
                raise ValueError(f"Missing expert.token_strings.{e}")
            to_add.append(token_str)
        tokenizer.add_tokens(to_add, special_tokens=False)

    # Precompute token ids and prefix ids (must be picklable for dataset.map).
    token_ids: Dict[str, int] = {}
    prefix_ids: Dict[str, List[int]] = {}

    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    for e in experts_norm:
        token_str = token_strs.get(e)
        if not token_str:
            raise ValueError(f"Missing expert.token_strings.{e}")
        token_id = _ensure_single_token(tokenizer, token_str)
        token_ids[e] = token_id

        k = _as_int(slots.get(e), default=0)
        if k <= 0:
            raise ValueError(f"expert.slots.{e} must be > 0, got {k}")
        slots[e] = k

        prefix = prefix_text.get(e, f"because the {e} feature is ")
        prefix_ids[e] = tokenizer.encode(prefix, add_special_tokens=False)

        if e in teacher_dims:
            teacher_dims[e] = int(teacher_dims[e])

    return ExpertRuntime(
        enabled=True,
        experts=tuple(experts_norm),
        token_ids=token_ids,
        token_strs={e: token_strs[e] for e in experts_norm},
        slots={e: int(slots[e]) for e in experts_norm},
        prefix_ids=prefix_ids,
        newline_ids=newline_ids,
        teacher_mode=teacher_mode,
        teacher_cache_dir=str(teacher_cache_dir) if teacher_cache_dir else None,
        teacher_require=teacher_require,
        teacher_write_back=teacher_write_back,
        teacher_sources={e: teacher_sources.get(e) for e in experts_norm},
        teacher_dims={e: int(teacher_dims.get(e, 0)) for e in experts_norm},
        align_enabled=align_enabled,
        align_weight=float(align_weight),
        align_loss=align_loss,
        normalize_teacher=normalize_teacher,
        normalize_student=normalize_student,
    )
