import argparse
import inspect
import json
import os
import re
from types import MethodType
from typing import Any, Dict, List, Optional, Set

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from grpo_data import build_grpo_dataset
from qwen_ivtlr import IVTLR
from utils import set_seed

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception as exc:
    raise ImportError(
        "TRL is required for GRPO. Please install it first, e.g. `pip install trl`."
    ) from exc


def _build_ivtlr_wrapper(model, tokenizer, processor, model_path: str):
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    return IVTLR(
        base_causallm=model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id,
        visual_end_id=visual_end_id,
        model_path=model_path,
    )


def _detect_runtime_backend() -> str:
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _setup_runtime_device(verbose: bool = True) -> str:
    backend = _detect_runtime_backend()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if backend == "npu" and local_rank >= 0:
        torch.npu.set_device(local_rank)
    elif backend == "cuda" and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    if verbose:
        rank = os.environ.get("RANK", "0")
        world_size = os.environ.get("WORLD_SIZE", "1")
        print(
            f"[Runtime] backend={backend}, local_rank={local_rank}, "
            f"rank={rank}, world_size={world_size}, torch_npu={HAS_TORCH_NPU}"
        )
    return backend


def _make_ivtlr_trl_compatible(ivtlr_model: IVTLR, cfg: Optional[Dict[str, Any]] = None):
    """
    GRPOTrainer may call forward without labels/position_ids.
    Patch IVTLR instance so it can be used directly as the training model.
    """
    original_forward = ivtlr_model.forward
    cfg = cfg or {}
    gen_defaults = {
        "do_sample": bool(cfg.get("do_sample", False)),
        "temperature": float(cfg.get("temperature", 1.0)),
        "top_k": int(cfg.get("top_k", 0)),
        "top_p": float(cfg.get("top_p", 1.0)),
    }

    def trl_forward(self, input_ids=None, attention_mask=None, labels=None, position_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required for IVTLR forward.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = (
                torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

        if labels is None:
            # GRPO uses logits to compute policy gradients. Loss is ignored.
            labels = input_ids.clone()

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            **kwargs,
        )

    ivtlr_model.forward = MethodType(trl_forward, ivtlr_model)
    ivtlr_model.config = ivtlr_model.base_causallm.config
    ivtlr_model.generation_config = getattr(ivtlr_model.base_causallm, "generation_config", None)

    def _select_next_token(logits, do_sample, temperature, top_k, top_p):
        if not do_sample:
            return int(torch.argmax(logits, dim=-1).item())

        temp = max(float(temperature), 1e-5)
        scores = logits / temp

        k = int(top_k) if top_k is not None else 0
        if k > 0:
            k = min(k, scores.size(-1))
            topk_vals, _ = torch.topk(scores, k)
            kth = topk_vals[-1]
            scores = scores.masked_fill(scores < kth, float("-inf"))

        p = float(top_p) if top_p is not None else 1.0
        if 0.0 < p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_indices_to_remove, float("-inf"))

            probs = torch.softmax(sorted_scores, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or torch.sum(probs) <= 0:
                return int(torch.argmax(logits, dim=-1).item())
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return int(sorted_indices[sampled_idx].item())

        probs = torch.softmax(scores, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any() or torch.sum(probs) <= 0:
            return int(torch.argmax(logits, dim=-1).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def _generate_one(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        max_new_tokens,
        do_sample,
        temperature,
        top_k,
        top_p,
    ):
        tokens = input_ids[0].detach().tolist()
        current_ids = input_ids.clone()

        position_ids = torch.arange(
            0, current_ids.shape[1], dtype=torch.long, device=current_ids.device
        ).reshape(1, -1)

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids) if attention_mask is None else attention_mask,
            labels=current_ids.clone(),
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        next_token = _select_next_token(
            outputs.logits[0, -1, :],
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens.append(next_token)

        current_inputs_embeds = outputs.inputs_embeds
        current_attention_mask = torch.ones(
            (1, current_inputs_embeds.shape[1]), device=current_inputs_embeds.device
        )
        next_token_embedding = self.embedding(
            torch.tensor([[next_token]], device=current_inputs_embeds.device)
        )
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1
        )

        past_key_values = None
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if past_key_values is None:
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(
                    0, current_inputs_embeds.shape[1], dtype=torch.long, device=current_inputs_embeds.device
                ).reshape(1, -1)
            else:
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor(
                    [[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device
                )

            base_outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None,
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = base_outputs.past_key_values

            next_token = _select_next_token(
                base_outputs.logits[0, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens.append(next_token)

            next_token_embedding = self.embedding(
                torch.tensor([[next_token]], device=current_inputs_embeds.device)
            )
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1
            )

            if next_token == self.eos_token_id:
                break

        return torch.tensor(tokens, device=input_ids.device).view(1, -1)

    def trl_generate(self, input_ids=None, attention_mask=None, pixel_values=None, image_grid_thw=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required for IVTLR generation.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        do_sample = bool(kwargs.get("do_sample", gen_defaults["do_sample"]))
        temperature = float(kwargs.get("temperature", gen_defaults["temperature"]))
        top_k = int(kwargs.get("top_k", gen_defaults["top_k"]))
        top_p = float(kwargs.get("top_p", gen_defaults["top_p"]))
        max_new_tokens = int(kwargs.get("max_new_tokens", kwargs.get("max_completion_length", 16)))

        # Native IVTLR.generate only supports batch_size == 1.
        # For GRPO rollouts, run per-sample and pad to a common length.
        if input_ids.shape[0] == 1:
            out = _generate_one(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            return out.to(input_ids.device)

        single_outputs = []
        for i in range(input_ids.shape[0]):
            pv = pixel_values[i : i + 1] if pixel_values is not None else None
            igt = image_grid_thw[i : i + 1] if image_grid_thw is not None else None
            out_i = _generate_one(
                self,
                input_ids=input_ids[i : i + 1],
                attention_mask=attention_mask[i : i + 1],
                pixel_values=pv,
                image_grid_thw=igt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ).to(input_ids.device)
            single_outputs.append(out_i)

        max_len = max(x.shape[1] for x in single_outputs)
        pad_id = getattr(self.base_causallm.config, "pad_token_id", None)
        if pad_id is None or pad_id < 0:
            pad_id = self.eos_token_id

        padded = []
        for x in single_outputs:
            if x.shape[1] < max_len:
                pad = torch.full(
                    (1, max_len - x.shape[1]),
                    pad_id,
                    dtype=x.dtype,
                    device=input_ids.device,
                )
                x = torch.cat([x, pad], dim=1)
            padded.append(x)

        return torch.cat(padded, dim=0)

    ivtlr_model.generate = MethodType(trl_generate, ivtlr_model)
    ivtlr_model._ivtlr_gen_defaults = gen_defaults

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.base_causallm, "save_pretrained"):
            self.base_causallm.save_pretrained(save_directory, **kwargs)
        torch.save(self.state_dict(), os.path.join(save_directory, "ivtlr_state_dict.pt"))

    ivtlr_model.save_pretrained = MethodType(save_pretrained, ivtlr_model)

    def add_model_tags(self, tags):
        # TRL may tag models for metadata/logging; keep this a safe no-op fallback.
        if tags is None:
            return
        if hasattr(self.base_causallm, "add_model_tags"):
            self.base_causallm.add_model_tags(tags)
            return

        if not hasattr(self, "_model_tags"):
            self._model_tags = set()
        if isinstance(tags, str):
            self._model_tags.add(tags)
            return
        try:
            self._model_tags.update(str(x) for x in tags)
        except TypeError:
            self._model_tags.add(str(tags))

    ivtlr_model.add_model_tags = MethodType(add_model_tags, ivtlr_model)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {}
        if hasattr(self.base_causallm, "gradient_checkpointing_enable"):
            self.base_causallm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=kwargs
            )
        if hasattr(self.base_causallm, "config"):
            # Match HF behavior when gradient checkpointing is enabled.
            self.base_causallm.config.use_cache = False

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_causallm, "gradient_checkpointing_disable"):
            self.base_causallm.gradient_checkpointing_disable()
        if hasattr(self.base_causallm, "config"):
            self.base_causallm.config.use_cache = True

    ivtlr_model.gradient_checkpointing_enable = MethodType(
        gradient_checkpointing_enable, ivtlr_model
    )
    ivtlr_model.gradient_checkpointing_disable = MethodType(
        gradient_checkpointing_disable, ivtlr_model
    )
    ivtlr_model.supports_gradient_checkpointing = bool(
        getattr(ivtlr_model.base_causallm, "supports_gradient_checkpointing", True)
    )
    return ivtlr_model


def _load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_model_path(cfg: Dict[str, Any]) -> str:
    model_type = str(cfg.get("model_type", "2B")).upper()
    if model_type == "7B":
        return cfg["model_path_7b"]
    return cfg["model_path_2b"]


def _build_lora_config(cfg: Dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        r=int(cfg.get("lora_r", 64)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias=str(cfg.get("lora_bias", "none")),
        inference_mode=False,
    )


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "module", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise ValueError("Unsupported checkpoint payload type.")
    return payload


def _candidate_key_variants(key: str) -> List[str]:
    variants = set()
    variants.add(key)
    if key.startswith("module."):
        variants.add(key[len("module.") :])
    else:
        variants.add("module." + key)

    expanded = set()
    for k in variants:
        expanded.add(k)
        if not k.startswith("base_causallm."):
            expanded.add("base_causallm." + k)
    return list(expanded)


def _remap_state_dict_keys(
    source_sd: Dict[str, torch.Tensor], target_keys: Set[str]
) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in source_sd.items():
        matched = None
        for cand in _candidate_key_variants(key):
            if cand in target_keys:
                matched = cand
                break
        if matched is not None:
            remapped[matched] = value
    return remapped


def _load_sft_checkpoint_into_model(
    model, tokenizer, processor, cfg: Dict[str, Any], model_path: str
):
    ckpt_path = cfg.get("sft_ckpt_path", "")
    if not ckpt_path:
        raise ValueError("sft_ckpt_path is required for GRPO fine-tuning in this workflow.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"sft_ckpt_path not found: {ckpt_path}")

    # Adapter directory (preferred).
    if os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
        print(f"[CKPT] Loading PEFT adapter from directory: {ckpt_path}")
        if isinstance(model, PeftModel):
            model.load_adapter(ckpt_path, adapter_name="sft", is_trainable=True)
            model.set_adapter("sft")
            return _build_ivtlr_wrapper(model, tokenizer, processor, model_path)
        peft_model = PeftModel.from_pretrained(model, ckpt_path, is_trainable=True)
        return _build_ivtlr_wrapper(peft_model, tokenizer, processor, model_path)

    # Full fp32 state dict from SFT (e.g. epoch_x_full_model_fp32.pth).
    print(f"[CKPT] Loading full SFT state dict from: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    source_sd = _extract_state_dict(payload)

    ivtlr_wrapper = _build_ivtlr_wrapper(model, tokenizer, processor, model_path)

    target_keys = set(ivtlr_wrapper.state_dict().keys())
    remapped = _remap_state_dict_keys(source_sd, target_keys)
    if len(remapped) == 0:
        raise RuntimeError(
            "No parameters were matched while loading SFT checkpoint. "
            "Please verify the checkpoint source and model architecture."
        )
    incompatible = ivtlr_wrapper.load_state_dict(remapped, strict=False)
    print(
        f"[CKPT] Loaded state dict with remapping: "
        f"loaded={len(remapped)}, missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    return ivtlr_wrapper


def _to_supported_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(callable_obj)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _align_special_token_ids(model, tokenizer) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        tok_val = getattr(tokenizer, attr, None)
        if hasattr(model, "config"):
            cur = getattr(model.config, attr, None)
            if cur != tok_val:
                setattr(model.config, attr, tok_val)
                updates[attr] = tok_val
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            cur = getattr(gen_cfg, attr, None)
            if cur != tok_val:
                setattr(gen_cfg, attr, tok_val)
                updates[attr] = tok_val
    return updates


def _parse_valid_letters(raw_letters: Any) -> Set[str]:
    if raw_letters is None:
        return set()
    if isinstance(raw_letters, list):
        return {str(x).strip().upper() for x in raw_letters if str(x).strip()}
    if isinstance(raw_letters, str):
        return {x.strip().upper() for x in raw_letters.split(",") if x.strip()}
    return {str(raw_letters).strip().upper()}


def _completion_to_text(completion: Any) -> str:
    # Compatible with several TRL completion structures.
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return str(completion[0].get("content", completion[0]))
        return " ".join(str(x) for x in completion)
    return str(completion)


def _extract_answer_letter(text: str, valid_letters: Set[str]) -> Optional[str]:
    """
    Regex extraction aligned with existing SFT prompting/output convention:
    - prompt ends with "Answer:"
    - supervised target uses "Therefore, the answer is X"
    """
    if not text:
        return None

    pattern = (
        r"(?:Therefore,?\s*the\s+answer\s+is|the\s+answer\s+is|answer\s+is:?|Answer:)"
        r"\s*(?:Option)?\s*\(?([A-Za-z])\)?"
    )
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if matches:
        for m in reversed(matches):
            pred = m.upper()
            if not valid_letters or pred in valid_letters:
                return pred

    # Fallback: last standalone letter.
    candidates = re.findall(r"\b([A-Za-z])\b", text)
    for c in reversed(candidates):
        pred = c.upper()
        if not valid_letters or pred in valid_letters:
            return pred

    return None


def build_reward_functions(cfg: Dict[str, Any]):
    """
    ACC reward only (per user request).
    """
    acc_weight = float(cfg.get("reward_acc_weight", 1.0))
    acc_correct = float(cfg.get("reward_acc_correct", 1.0))
    acc_wrong = float(cfg.get("reward_acc_wrong", -0.2))
    acc_missing = float(cfg.get("reward_acc_missing", -0.4))
    verbose_rollout = bool(cfg.get("verbose_rollout_print", False))
    verbose_max_chars = int(cfg.get("verbose_rollout_max_chars", 300))
    print_only_rank0 = bool(cfg.get("verbose_rollout_rank0_only", True))

    def _should_print():
        if not verbose_rollout:
            return False
        if not print_only_rank0:
            return True
        rank = os.environ.get("RANK")
        if rank is None:
            return True
        try:
            return int(rank) == 0
        except Exception:
            return True

    def _truncate(text: str) -> str:
        if len(text) <= verbose_max_chars:
            return text
        return text[:verbose_max_chars] + "...<truncated>"

    def accuracy_reward(completions, ground_truth=None, valid_letters=None, **kwargs):
        rewards = []
        ground_truth = ground_truth or kwargs.get("answer") or kwargs.get("answers")
        if ground_truth is None:
            raise ValueError("ground_truth column is required for accuracy reward.")
        if valid_letters is None:
            valid_letters = [None] * len(completions)

        prompt_list = (
            kwargs.get("prompt")
            or kwargs.get("prompts")
            or kwargs.get("question")
            or kwargs.get("questions")
        )

        for completion, gt, letters in zip(completions, ground_truth, valid_letters):
            text = _completion_to_text(completion)
            gt_letter = str(gt).strip().upper()
            valid = _parse_valid_letters(letters)
            pred = _extract_answer_letter(text, valid)
            if pred is None:
                rewards.append(acc_missing * acc_weight)
            elif pred == gt_letter:
                rewards.append(acc_correct * acc_weight)
            else:
                rewards.append(acc_wrong * acc_weight)

        if _should_print():
            # Log each question rollout in current reward call.
            for i, (completion, gt, letters, reward) in enumerate(
                zip(completions, ground_truth, valid_letters, rewards)
            ):
                text = _completion_to_text(completion)
                gt_letter = str(gt).strip().upper()
                valid = _parse_valid_letters(letters)
                pred = _extract_answer_letter(text, valid)
                prompt_i = ""
                if isinstance(prompt_list, list) and i < len(prompt_list):
                    prompt_i = str(prompt_list[i])
                elif isinstance(prompt_list, str):
                    prompt_i = prompt_list

                if prompt_i:
                    print(
                        "[ROLLOUT]"
                        f" idx={i}"
                        f" gt={gt_letter}"
                        f" pred={pred}"
                        f" reward={reward:.4f}"
                        f" prompt={_truncate(prompt_i)}"
                    )
                else:
                    print(
                        "[ROLLOUT]"
                        f" idx={i}"
                        f" gt={gt_letter}"
                        f" pred={pred}"
                        f" reward={reward:.4f}"
                    )
                print(f"[ROLLOUT_COMPLETION] { _truncate(text) }")
        return rewards

    return [accuracy_reward]


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL GRPO training entry")
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config_file)
    set_seed(int(cfg.get("seed", 0)))
    runtime_backend = _setup_runtime_device(verbose=True)

    output_dir = cfg.get("output_dir", "./output_grpo")
    os.makedirs(output_dir, exist_ok=True)

    model_path = _pick_model_path(cfg)
    print(f"[Init] model_path={model_path}")

    # NPU usually works more reliably with eager attention in this workflow.
    attn_impl = cfg.get("attn_implementation", "eager")
    if runtime_backend == "npu" and attn_impl != "eager":
        print(
            f"[Init] Overriding attn_implementation from {attn_impl} to eager for NPU runtime."
        )
        attn_impl = "eager"

    torch_dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) else torch.float16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Keep latent/special tokens aligned with SFT pipeline.
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    processor = AutoProcessor.from_pretrained(model_path, tokenizer=tokenizer)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.pad_token = tokenizer.pad_token
        processor.tokenizer.padding_side = tokenizer.padding_side

    model.resize_token_embeddings(len(tokenizer))
    token_id_updates = _align_special_token_ids(model, tokenizer)
    if token_id_updates:
        print(f"[Init] Aligned model token ids with tokenizer: {token_id_updates}")
    if bool(cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if bool(cfg.get("use_lora", True)):
        lora_cfg = _build_lora_config(cfg)
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model = _load_sft_checkpoint_into_model(model, tokenizer, processor, cfg, model_path)
    model = _make_ivtlr_trl_compatible(model, cfg)
    print("[Init] Using IVTLR model for GRPO training.")

    # Accept legacy key name `ksample` while preferring `k_samples`.
    k_samples_cfg = cfg.get("k_samples", cfg.get("ksample", 128))
    train_dataset = build_grpo_dataset(
        dataset_name=str(cfg.get("dataset_name", "m3cot")),
        processor=processor,
        k_samples=int(k_samples_cfg),
        seed=int(cfg.get("seed", 0)),
        max_latent_stage=int(cfg.get("max_latent_stage", 5)),
        pad_latent_to_max=bool(cfg.get("pad_latent_to_max", True)),
        num_proc=int(cfg.get("num_proc", 1)),
    )
    print(f"[Data] dataset_size={len(train_dataset)}")

    reward_funcs = build_reward_functions(cfg)

    grpo_kwargs = {
        "output_dir": output_dir,
        "run_name": cfg.get("run_name", "qwen2vl_grpo"),
        "seed": int(cfg.get("seed", 0)),
        "learning_rate": float(cfg.get("learning_rate", 1e-6)),
        "lr_scheduler_type": str(cfg.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.03)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 8)),
        "logging_steps": int(cfg.get("logging_steps", 1)),
        "save_steps": int(cfg.get("save_steps", 50)),
        "save_total_limit": int(cfg.get("save_total_limit", 3)),
        "bf16": bool(cfg.get("bf16", True)),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "report_to": cfg.get("report_to", "none"),
        "max_prompt_length": cfg.get("max_prompt_length", None),
        "max_completion_length": int(cfg.get("max_completion_length", 256)),
        "num_generations": int(cfg.get("num_generations", 4)),
        "do_sample": bool(cfg.get("do_sample", False)),
        "temperature": float(cfg.get("temperature", 1.0)),
        "top_k": int(cfg.get("top_k", 0)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "beta": float(cfg.get("beta", 0.0)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 0)),
        "log_completions": bool(cfg.get("log_completions", True)),
    }
    grpo_cfg = GRPOConfig(**_to_supported_kwargs(GRPOConfig, grpo_kwargs))

    # ---------------------------------------------------------------------------
    # Custom data collator: restore prompt from JSON string back to list[dict]
    # so that TRL's is_conversational() check passes for multimodal GRPO.
    # ---------------------------------------------------------------------------
    def _grpo_collator(features):
        for f in features:
            if isinstance(f.get("prompt"), str):
                f["prompt"] = json.loads(f["prompt"])
        return features

    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": grpo_cfg,
        "train_dataset": train_dataset,
    }
    trainer_init_sig = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in trainer_init_sig.parameters:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_init_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)

    # Override the default identity collator with ours.
    trainer.data_collator = _grpo_collator

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[Done] GRPO finished. Artifacts saved to: {final_dir}")


if __name__ == "__main__":
    main()
