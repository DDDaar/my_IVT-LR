import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import logging
logging.basicConfig(
    filename='qwenvl_32_infer_sqa_time_epoch4.log',
    level=logging.DEBUG,         
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)
import pdb
from transformers.cache_utils import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 4




class IVTLR(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        image_token_id,
        visual_start_id,
        visual_end_id,
        num_selected_patches: int = 32,
        model_path: str = None,  # [新增参数]
        use_head_gate: bool = False,
        candidate_pool_ratio: float = 1.0,
        candidate_pool_max: int = 64,
        w_attn: float = 1.0,
        w_grad: float = 0.0,
        w_delta: float = 0.0,
        score_norm: str = "minmax",
        grad_norm_type: str = "l2",
        grad_probe_mode: str = "answer_option",
        delta_probe_tokens: int = 8,
        delta_mask_mode: str = "zero",
        delta_topm_cap: int = 24,
        fallback_to_attn_when_invalid: bool = True,
    ):

        super(IVTLR, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.image_token_id = image_token_id
        self.visual_start_id = visual_start_id
        self.visual_end_id = visual_end_id
        self.num_selected_patches = num_selected_patches
        self.use_head_gate = bool(use_head_gate)
        self.candidate_pool_ratio = float(candidate_pool_ratio)
        self.candidate_pool_max = int(candidate_pool_max)
        self.w_attn = float(w_attn)
        self.w_grad = float(w_grad)
        self.w_delta = float(w_delta)
        self.score_norm = str(score_norm).lower()
        self.grad_norm_type = str(grad_norm_type).lower()
        self.grad_probe_mode = str(grad_probe_mode).lower()
        self.delta_probe_tokens = int(delta_probe_tokens)
        self.delta_mask_mode = str(delta_mask_mode).lower()
        self.delta_topm_cap = int(delta_topm_cap)
        self.fallback_to_attn_when_invalid = bool(fallback_to_attn_when_invalid)
        # Trace is disabled by default to keep original behavior unchanged.
        self.enable_trace = False
        self.last_trace = []
        # Optional per-layer raw-attention trace for visualization; disabled by default.
        self.trace_save_raw_layer_scores = False
        self.trace_layer_indices = None
        print(f'选择了{num_selected_patches}个 patch')

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        # [修改] 使用传入的 model_path，如果未传入则使用默认值 (可选)
        if model_path is None:
            model_path = "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct"
            import time
            print('no model path!!!')
            time.sleep(1000)
            
        self.processor = AutoProcessor.from_pretrained(model_path)
        # self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
        
        #####################################################################
        # #增加全连接层进行注意力的融合，而不是简单平均各个head
        # #1. 获取 num_heads
        # # 大多数 HF 模型（Qwen2-VL, Chameleon, Llama）使用 .num_attention_heads
        # # GPT-2 使用 .n_head
        # if hasattr(self.base_causallm.config, "num_attention_heads"):
        #     num_heads = self.base_causallm.config.num_attention_heads
        # elif hasattr(self.base_causallm.config, "n_head"):
        #     num_heads = self.base_causallm.config.n_head
        # else:
        #     raise ValueError("Cannot find number of attention heads in model config")
            
        # self.head_fusion = nn.Sequential(
        #     nn.Linear(num_heads, 1), # 将多头权重融合为1个分数
        #     nn.Sigmoid()
        # )
        # print('使用mlp层进行head注意力融合')
        ####################################################################

#         ####################################################################
        # #魔改2、3
        # 仅在 use_head_gate=True 时构建 gate，避免默认路径依赖特定 config 字段。
        self.head_gate = None
        if self.use_head_gate:
            def _pick(cfg, names):
                if cfg is None:
                    return None
                for n in names:
                    if hasattr(cfg, n):
                        v = getattr(cfg, n)
                        if v is not None:
                            return v
                return None

            cfgs = []
            root_cfg = getattr(self.base_causallm, "config", None)
            cfgs.append(root_cfg)
            cfgs.append(getattr(root_cfg, "text_config", None))

            language_model = getattr(self.base_causallm, "language_model", None)
            lm_cfg = getattr(language_model, "config", None)
            cfgs.append(lm_cfg)
            cfgs.append(getattr(lm_cfg, "text_config", None))

            base_model = getattr(self.base_causallm, "base_model", None)
            inner_model = getattr(base_model, "model", None)
            inner_cfg = getattr(inner_model, "config", None)
            cfgs.append(inner_cfg)
            cfgs.append(getattr(inner_cfg, "text_config", None))

            num_heads = None
            hidden_size = None
            for cfg in cfgs:
                if num_heads is None:
                    num_heads = _pick(cfg, ["num_attention_heads", "n_head"])
                if hidden_size is None:
                    hidden_size = _pick(cfg, ["hidden_size", "n_embd", "d_model"])
                if num_heads is not None and hidden_size is not None:
                    break

            if num_heads is None or hidden_size is None:
                raise ValueError(
                    "Cannot find head-gate dims in model config. "
                    "Please set use_head_gate=false or check model config fields."
                )

            self.head_gate = nn.Sequential(
                nn.Linear(int(hidden_size), int(num_heads)), # 输入 latent hidden state，输出每个 head 的权重
                nn.Softmax(dim=-1) # 保证权重和为 1
            )
#         ####################################################################

    
        ####################################################################
#         #魔改2plus——layer间也动态权重
#         #1. 获取 num_heads
#         # 大多数 HF 模型（Qwen2-VL, Chameleon, Llama）使用 .num_attention_heads
#         # GPT-2 使用 .n_head
#         if hasattr(self.base_causallm.config, "num_attention_heads"):
#             num_heads = self.base_causallm.config.num_attention_heads
#         elif hasattr(self.base_causallm.config, "n_head"):
#             num_heads = self.base_causallm.config.n_head
#         else:
#             raise ValueError("Cannot find number of attention heads in model config")


#         if hasattr(self.base_causallm.config, "hidden_size"):
#             hidden_size = self.base_causallm.config.hidden_size
#         elif hasattr(self.base_causallm.config, "n_embd"):
#             hidden_size = self.base_causallm.config.n_embd
            
            
#                 # 自动检测层数
#         if hasattr(self.base_causallm.config, "num_hidden_layers"):
#             num_layers = self.base_causallm.config.num_hidden_layers
#         elif hasattr(self.base_causallm.config, "n_layer"): # 某些旧模型或 GPT-2 风格模型使用 n_layer
#             num_layers = self.base_causallm.config.n_layer
        
#         self.head_gate = nn.Sequential(
#             nn.Linear(hidden_size, num_heads), # 输入 latent hidden state，输出每个 head 的权重
#             nn.Softmax(dim=-1) # 保证权重和为 1
#         )
        
#         self.layer_gate = nn.Sequential(
#               nn.Linear(hidden_size, num_layers),  # num_layers = len(attentions)，Qwen2-VL 7B=28
#               nn.Softmax(dim=-1)
#         )
        ####################################################################
    
    
    
        ####################################################################
        # #魔改4
        # # ---------- 替换掉之前的 head_gate 逻辑 ----------
        # if hasattr(self.base_causallm.config, "hidden_size"):
        #     hidden_size = self.base_causallm.config.hidden_size
        # elif hasattr(self.base_causallm.config, "n_embd"):
        #     hidden_size = self.base_causallm.config.n_embd
        # else:
        #     raise ValueError("Cannot find hidden size in model config")
            
        # # 引入可学习的视觉选择器 (交叉注意力)
        # # 降维以减少计算量并防止过拟合，例如降至 hidden_size 的 1/4
        # embed_dim = hidden_size // 4 
        # self.visual_q_proj = nn.Linear(hidden_size, embed_dim)
        # self.visual_k_proj = nn.Linear(hidden_size, embed_dim)
        # self.temperature = embed_dim ** 0.5
        # # ------------------------------------------------
        ####################################################################
        
        
#         ####################################################################
#         #魔改5，直接融合成新的视觉特征，而不是复用输入的embedding
#         #1. 获取 num_heads
#         # 大多数 HF 模型（Qwen2-VL, Chameleon, Llama）使用 .num_attention_heads
#         # GPT-2 使用 .n_head
#         if hasattr(self.base_causallm.config, "num_attention_heads"):
#             num_heads = self.base_causallm.config.num_attention_heads
#         elif hasattr(self.base_causallm.config, "n_head"):
#             num_heads = self.base_causallm.config.n_head
#         else:
#             raise ValueError("Cannot find number of attention heads in model config")
            
#         if hasattr(self.base_causallm.config, "hidden_size"):
#             hidden_size = self.base_causallm.config.hidden_size
#         elif hasattr(self.base_causallm.config, "n_embd"):
#             hidden_size = self.base_causallm.config.n_embd

        
#         # 初始化 K 个可学习的 Query Tokens (代表你想提取的 K 个视觉特征)
#         self.num_selected_patches = num_selected_patches
#         self.visual_queries = nn.Parameter(torch.randn(1, self.num_selected_patches, hidden_size))
#         # 定义一个小型的 Cross-Attention 模块
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
#         self.visual_proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, hidden_size)
#         )
#         ####################################################################
        
        
    def _normalize_scores(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if self.score_norm == "zscore":
            mu = x.mean()
            sigma = x.std(unbiased=False)
            if torch.isnan(sigma) or sigma <= 1e-8:
                return torch.zeros_like(x)
            return (x - mu) / (sigma + 1e-8)
        if self.score_norm == "softmax":
            return torch.softmax(x, dim=-1)
        x_min = x.min()
        x_max = x.max()
        denom = x_max - x_min
        if torch.isnan(denom) or denom <= 1e-8:
            return torch.zeros_like(x)
        return (x - x_min) / (denom + 1e-8)

    def _get_probe_token_ids_from_labels(self, labels_row: torch.Tensor):
        valid = labels_row[labels_row != -100]
        if valid.numel() == 0:
            return [int(self.eos_token_id)]

        valid_list = [int(x) for x in valid.detach().cpu().tolist()]
        tokenizer = self.processor.tokenizer
        anchor = tokenizer.encode("Therefore, the answer is ", add_special_tokens=False)

        answer_tokens = []
        if anchor and len(valid_list) >= len(anchor) + 1:
            for i in range(len(valid_list) - len(anchor)):
                if valid_list[i : i + len(anchor)] == anchor:
                    start = i + len(anchor)
                    end = min(len(valid_list), start + max(1, self.delta_probe_tokens))
                    answer_tokens = [int(x) for x in valid_list[start:end]]
                    break

        mode = self.grad_probe_mode
        if mode == "answer_span" and answer_tokens:
            probe_ids = answer_tokens
        elif mode == "last_token":
            probe_ids = [int(valid_list[-1])]
        elif mode == "first_token":
            probe_ids = [int(valid_list[0])]
        else:
            if answer_tokens:
                probe_ids = [int(answer_tokens[0])]
            else:
                probe_ids = [int(valid_list[0])]

        dedup = []
        seen = set()
        for tid in probe_ids:
            if tid not in seen:
                dedup.append(int(tid))
                seen.add(int(tid))
        return dedup if dedup else [int(self.eos_token_id)]

    def _compute_grad_scores(
        self,
        logits_this: torch.Tensor,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor,
        abs_cands: torch.Tensor,
        b: int,
        end: int,
    ) -> torch.Tensor:
        if abs_cands.numel() == 0:
            return torch.zeros_like(abs_cands, dtype=inputs_embeds.dtype)

        probe_ids = self._get_probe_token_ids_from_labels(labels[b])
        probe_pos = max(0, min(end - 1, logits_this.size(1) - 1))
        probe_terms = [logits_this[b, probe_pos, int(pid)] for pid in probe_ids]
        probe_scalar = torch.stack(probe_terms).mean()
        grad = torch.autograd.grad(
            probe_scalar,
            inputs_embeds,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return torch.zeros(abs_cands.numel(), device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        cand_grad = grad[b, abs_cands, :]
        if self.grad_norm_type == "l1":
            scores = cand_grad.abs().sum(dim=-1)
        else:
            scores = cand_grad.norm(p=2, dim=-1)
        return scores.detach()

    def _compute_delta_scores(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        logits_this: torch.Tensor,
        labels: torch.Tensor,
        abs_cands: torch.Tensor,
        b: int,
        end: int,
    ) -> torch.Tensor:
        if abs_cands.numel() == 0:
            return torch.zeros_like(abs_cands, dtype=inputs_embeds.dtype)

        probe_ids = self._get_probe_token_ids_from_labels(labels[b])
        probe_pos = max(0, min(end - 1, logits_this.size(1) - 1))
        probe_ids_t = torch.tensor(probe_ids, device=logits_this.device, dtype=torch.long)
        base_logp = F.log_softmax(logits_this[b, probe_pos, :], dim=-1).index_select(0, probe_ids_t).mean().detach()

        max_eval = min(abs_cands.numel(), max(0, self.delta_topm_cap))
        scores = torch.zeros(abs_cands.numel(), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        if max_eval == 0:
            return scores

        mean_vec = None
        if self.delta_mask_mode == "mean":
            mean_vec = inputs_embeds[b, :end, :].mean(dim=0, keepdim=True)

        for i in range(max_eval):
            abs_idx = int(abs_cands[i].item())
            if abs_idx >= end:
                continue

            with torch.no_grad():
                cf_embeds = inputs_embeds[b : b + 1, :end, :].detach().clone()
                if self.delta_mask_mode == "mean" and mean_vec is not None:
                    cf_embeds[0, abs_idx, :] = mean_vec[0]
                else:
                    cf_embeds[0, abs_idx, :] = 0.0

                cf_outputs = self.base_causallm(
                    inputs_embeds=cf_embeds,
                    attention_mask=attention_mask[b : b + 1, :end],
                    position_ids=position_ids[b : b + 1, :end],
                    pixel_values=pixel_values[b : b + 1] if pixel_values is not None else None,
                    image_grid_thw=image_grid_thw[b : b + 1] if image_grid_thw is not None else None,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=False,
                )
                cf_logp = F.log_softmax(cf_outputs.logits[0, probe_pos, :], dim=-1).index_select(0, probe_ids_t).mean()
                scores[i] = (base_logp - cf_logp).clamp_min(0.0)

        return scores

    def _fuse_and_select(
        self,
        rel_scores: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        logits_this: torch.Tensor,
        labels: torch.Tensor,
        b: int,
        end: int,
        vs: int,
    ):
        k = int(self.num_selected_patches)
        if rel_scores.numel() == 0 or k <= 0:
            empty = torch.zeros(0, dtype=torch.long, device=rel_scores.device)
            empty_f = torch.zeros(0, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            return empty, empty, empty_f, empty_f, empty_f

        k = min(k, rel_scores.numel())
        cand_m = max(k, int(round(k * max(self.candidate_pool_ratio, 1.0))))
        cand_m = min(cand_m, rel_scores.numel())
        if int(self.candidate_pool_max) > 0:
            cand_m = min(cand_m, int(self.candidate_pool_max))
        cand_m = max(k, cand_m)

        cand_attn_scores, cand_rel = rel_scores.topk(cand_m, sorted=False)
        cand_abs = (vs + 1) + cand_rel

        grad_scores = torch.zeros_like(cand_attn_scores)
        if self.w_grad != 0.0:
            grad_scores = self._compute_grad_scores(
                logits_this=logits_this,
                inputs_embeds=inputs_embeds,
                labels=labels,
                abs_cands=cand_abs,
                b=b,
                end=end,
            )

        delta_scores = torch.zeros_like(cand_attn_scores)
        if self.w_delta != 0.0:
            delta_scores = self._compute_delta_scores(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                logits_this=logits_this,
                labels=labels,
                abs_cands=cand_abs,
                b=b,
                end=end,
            )

        fused = (
            self.w_attn * self._normalize_scores(cand_attn_scores)
            + self.w_grad * self._normalize_scores(grad_scores)
            + self.w_delta * self._normalize_scores(delta_scores)
        )

        invalid = torch.isnan(fused) | torch.isinf(fused)
        if invalid.any():
            if self.fallback_to_attn_when_invalid:
                fused = cand_attn_scores
            else:
                fused = torch.where(invalid, torch.full_like(fused, float("-inf")), fused)

        top_idx = fused.topk(k, sorted=False).indices
        top_rel = cand_rel[top_idx]
        top_abs = cand_abs[top_idx]
        top_scores = fused[top_idx]
        return top_rel, top_abs, top_scores, cand_attn_scores, grad_scores

    def forward(
        self,
        input_ids: torch.LongTensor,        # shape = (B, S)
        attention_mask: torch.LongTensor,    # shape = (B, S)
        labels: torch.LongTensor,            # shape = (B, S)
        position_ids: torch.LongTensor,      # shape = (B, S)
        pixel_values: torch.FloatTensor,     # shape = (B, 3, H, W)
        image_grid_thw: torch.Tensor = None,
        **kwargs
    ):

        B, S = input_ids.size()

        # decode
        _ = self.processor.tokenizer.batch_decode(
            input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        # 将输入的token id转为embeddings
        inputs_embeds = self.embedding(input_ids)  # (B, S, D)

        original_mask = torch.ones((B, S), dtype=torch.bool, device=input_ids.device)

        vs_indices = (input_ids == self.visual_start_id).nonzero(as_tuple=True)
        # vs_indices = (tensor([0, 1]), tensor([2, 3]))
        # 第一个tensor是批次索引：[批次0, 批次1]
        # 第二个tensor是序列位置：[位置2, 位置3]
        ve_indices = (input_ids == self.visual_end_id).nonzero(as_tuple=True)
#     vs_pos_per_batch = {
#     0: 2,  # 批次0的视觉开始标记在位置2
#     1: 3   # 批次1的视觉开始标记在位置3
# }
        vs_pos_per_batch = {b.item(): vs_indices[1][i].item() for i, b in enumerate(vs_indices[0])}
        ve_pos_per_batch = {b.item(): ve_indices[1][i].item() for i, b in enumerate(ve_indices[0])}

        
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.base_causallm.visual.get_dtype())
            image_embeds = self.base_causallm.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_embeds.shape[0]}"
                )
                
            #print(f'当前序列中，图像 token 有{n_image_tokens}个')
            
            # 图像部分掩码
            image_mask_init = (input_ids == self.image_token_id)  # (B, orig_S)
            # 假设D=768，则expand_mask形状为(B, S, 768)，每个图像token位置的所有768维都是True
            expand_mask = image_mask_init.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1))
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            #替换图像token的嵌入
            inputs_embeds = inputs_embeds.masked_scatter(expand_mask, image_embeds)
        else:
            image_mask_init = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        
        #将所有序列统一到最大长度，方便批处理
        max_len = 4000
        image_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)
        image_mask[:, :S] = image_mask_init


        for b in range(B):
            #标记出每一条数据（Batch）中图像 Token 所在的索引范围。
            vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
            image_mask[b, vs+1:ve] = True

        #latent token的位置
        #max_n_latents 计算当前 Batch 中最长的推理链条长度
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == b]
            for b in range(B)
        ]
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        # 如果存在推理 Token，它将 end 设为 第一个思考 Token 出现的位置。
        # 这意味着模型在处理输入时，只会一次性处理到第一个 <|thought|> 之前的内容（通常是图像和问题提示词）。
        if max_n_latents > 0:
            first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0)
            end = first_latent_pos
        else:
            end = S
        
        kv_cache = None
        all_logits = []

        #外层循环：多轮潜变量处理
        if max_n_latents > 0:
            #print(f'max_n_latents={max_n_latents}')
            # 对k个latent token依次处理
            for pass_idx in range(max_n_latents):
                #初始化每轮的变量，从头开始
                start = 0
                hidden_states_offset = 0
                #无KV缓存：attention_mask只取当前段(start:end)
                #有KV缓存：attention_mask需要包含之前的所有token(:end)
                if kv_cache is None:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],  # (B, end, D)
                        attention_mask=attention_mask[:, start:end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )

                logits_this = outputs.logits     #当前步的logits    
                
                # --- 【新增检查点 2】 ---
                if torch.isnan(logits_this).any():
                    print(f"🚨 致命: 在 pass_idx {pass_idx}，LLM 前向传播输出的 logits 变成了 NaN!")
                    print("这说明你拼接进去的图像特征导致了 LLM 内部算力溢出 (通常在 LayerNorm 或 Softmax 层)。")
                    #import pdb; pdb.set_trace()
                # -----------------------
                
                hidden_states = outputs.hidden_states[-1]     # 最后一层的隐藏状态 
                
                all_hidden_states = outputs.hidden_states  # 所有层的隐藏状态
                
                
                
                attentions    = outputs.attentions        #所有注意力层的注意力权重列表 list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)

#######################################################################原始                
                #   Top-K
               #  avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len) 将所有层的注意力矩阵在 heads（头）维度上拼接，(B, L * heads, seq_len, seq_len)---->(B, seq_len, seq_len)
               #  current_seq_len = avg_attn.size(1) #seq长度

               #  select_image_embeds = []

               #  for b in range(B):
               #      #最后一个位置的注意力图
               #      last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
               #      vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
               #      scores = last_attn.clone()
                    
               #      allowed_positions = image_mask[b, :current_seq_len]  # shape=(S,)
               #      invalid = ~allowed_positions
               #      #将非图像位置的分数设为负无穷，确保不会被选中
               #      scores[invalid] = float("-inf")

               #      rel_scores = scores[vs+1 : ve]  # (image_len,)
               #      #选择图像token中的topk个
               #      topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
               #      abs_idxs = (vs + 1) + topk_rel
               #      if self.enable_trace:
               #          topk_scores = rel_scores[topk_rel]
               #          grid_thw_b = None
               #          if image_grid_thw is not None:
               #              grid_row = image_grid_thw[b]
               #              if torch.is_tensor(grid_row):
               #                  grid_thw_b = [int(x) for x in grid_row.detach().cpu().tolist()]
               #          self.last_trace.append(
               #              {
               #                  "pass_idx": int(pass_idx),
               #                  "batch_idx": int(b),
               #                  "vs": int(vs),
               #                  "ve": int(ve),
               #                  "rel_len": int(rel_scores.numel()),
               #                  "topk_rel": [int(x) for x in topk_rel.detach().cpu().tolist()],
               #                  "topk_abs": [int(x) for x in abs_idxs.detach().cpu().tolist()],
               #                  "topk_scores": [float(x) for x in topk_scores.detach().float().cpu().tolist()],
               #                  "rel_scores": [float(x) for x in rel_scores.detach().float().cpu().tolist()],
               #                  "grid_thw": grid_thw_b,
               #              }
               #          )
               #      logging.debug(f"topk_rel: {topk_rel}")
               #      logging.debug(f"abs idx: {abs_idxs}")
               #      image_mask[b, abs_idxs] = False

               #      #提取对应位置的embedding
               #      picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
               #      select_image_embeds.append(picked)
               # #截止到'避免梯度传播'前面
                    
#######################################################################原始   
                
                
################################################################################
#                 # 使用刚加的模块：mlp选择，魔改1
#                 # --- [修改开始] 使用 head_fusion 层进行融合 ---
                
#                 # 1. 对每一层的注意力矩阵应用 head_fusion
#                 layer_fused_attns = []
#                 for layer_attn in attentions:
#                     # layer_attn shape: (B, num_heads, S, S)
                    
#                     # 调整维度，将 num_heads 放到最后，以便 Linear 层处理
#                     # permute -> (B, S, S, num_heads)
#                     layer_attn_perm = layer_attn.permute(0, 2, 3, 1)
                    
#                     # 应用你的 head_fusion (Linear + Sigmoid)
#                     # 输入: (..., num_heads) -> 输出: (..., 1)
#                     # 结果 shape: (B, S, S, 1)
#                     fused_score = self.head_fusion(layer_attn_perm)
                    
#                     # 去掉最后一维 -> (B, S, S)
#                     layer_fused_attns.append(fused_score.squeeze(-1))

#                 # 2. 将各个层融合 (这里采用层间平均)
#                 # stack -> (num_layers, B, S, S)
#                 # mean(dim=0) -> (B, S, S)
#                 avg_attn = torch.stack(layer_fused_attns, dim=0).mean(dim=0)
                
#                 # --- [修改结束] ---

#                 current_seq_len = avg_attn.size(1)

#                 select_image_embeds = []

#                 for b in range(B):
#                     #最后一个位置的注意力图
#                     last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
#                     vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
#                     scores = last_attn.clone()
                    
#                     allowed_positions = image_mask[b, :current_seq_len]  # shape=(S,)
#                     invalid = ~allowed_positions
#                     #将非图像位置的分数设为负无穷，确保不会被选中
#                     scores[invalid] = float("-inf")

#                     rel_scores = scores[vs+1 : ve]  # (image_len,)
#                     #选择图像token中的topk个
#                     topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
#                     abs_idxs = (vs + 1) + topk_rel
#                     logging.debug(f"topk_rel: {topk_rel}")
#                     logging.debug(f"abs idx: {abs_idxs}")
#                     image_mask[b, abs_idxs] = False

#                     #提取对应位置的embedding
#                     picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
#                     select_image_embeds.append(picked)
###############################################################################


################################################################################
# # 使用hiddenstate->head提取各个head的权重进行加权求和，魔改2
                # 在 forward 循环中修改 (约 175 行附近)
                # hidden_states: (B, Seq_Len, Hidden_Size)
                # 我们关注的是产生 Attention 的那个 Latent Token，即位置 end-1
                # Selection path: raw attention top-k by default (forward-compatible),
                # with optional head-gate weighted attention.
                if self.use_head_gate:
                    current_latent_vector = hidden_states[:, end - 1, :]  # (B, Hidden_Size)
                    dynamic_head_weights = self.head_gate(current_latent_vector)
                    dynamic_head_weights = dynamic_head_weights.unsqueeze(-1).unsqueeze(-1)

                    layer_fused_attns = []
                    for layer_attn in attentions:
                        weighted_attn = (layer_attn * dynamic_head_weights).sum(dim=1)
                        layer_fused_attns.append(weighted_attn)
                    avg_attn = torch.stack(layer_fused_attns, dim=0).mean(dim=0)
                else:
                    avg_attn = torch.cat(attentions, dim=1).mean(dim=1)

                current_seq_len = avg_attn.size(1)
                select_image_embeds = []

                for b in range(B):
                    last_attn = avg_attn[b, end - 1]
                    vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    scores = last_attn.clone()

                    allowed_positions = image_mask[b, :current_seq_len]
                    invalid = ~allowed_positions
                    scores[invalid] = float("-inf")

                    rel_scores = scores[vs + 1 : ve]
                    topk_rel, abs_idxs, topk_scores, cand_attn_scores, cand_grad_scores = self._fuse_and_select(
                        rel_scores=rel_scores,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        logits_this=logits_this,
                        labels=labels,
                        b=b,
                        end=end,
                        vs=vs,
                    )

                    if self.enable_trace:
                        grid_thw_b = None
                        if image_grid_thw is not None:
                            grid_row = image_grid_thw[b]
                            if torch.is_tensor(grid_row):
                                grid_thw_b = [int(x) for x in grid_row.detach().cpu().tolist()]
                            else:
                                grid_thw_b = [int(x) for x in grid_row]

                        trace_payload = {
                            "pass_idx": int(pass_idx),
                            "batch_idx": int(b),
                            "vs": int(vs),
                            "ve": int(ve),
                            "rel_len": int(rel_scores.numel()),
                            "topk_rel": [int(x) for x in topk_rel.detach().cpu().tolist()],
                            "topk_abs": [int(x) for x in abs_idxs.detach().cpu().tolist()],
                            "topk_scores": [float(x) for x in topk_scores.detach().float().cpu().tolist()],
                            "rel_scores": [float(x) for x in rel_scores.detach().float().cpu().tolist()],
                            "candidate_attn_scores": [float(x) for x in cand_attn_scores.detach().float().cpu().tolist()],
                            "candidate_grad_scores": [float(x) for x in cand_grad_scores.detach().float().cpu().tolist()],
                            "grid_thw": grid_thw_b,
                        }

                        if self.trace_save_raw_layer_scores and self.trace_layer_indices:
                            num_layers = len(attentions)
                            normalized_layers = []
                            seen = set()
                            for raw_idx in self.trace_layer_indices:
                                idx = int(raw_idx)
                                if idx < 0:
                                    idx = num_layers + idx
                                if idx < 0 or idx >= num_layers:
                                    continue
                                if idx in seen:
                                    continue
                                seen.add(idx)
                                normalized_layers.append(idx)

                            raw_layer_rel_scores = []
                            for layer_idx in normalized_layers:
                                layer_attn = attentions[layer_idx]
                                layer_last_attn = layer_attn[b, :, end - 1].mean(dim=0)
                                layer_scores = layer_last_attn.clone()
                                layer_scores[invalid] = float("-inf")
                                layer_rel_scores = layer_scores[vs + 1 : ve]
                                raw_layer_rel_scores.append(
                                    [float(x) for x in layer_rel_scores.detach().float().cpu().tolist()]
                                )

                            if normalized_layers:
                                trace_payload["raw_layer_indices"] = [int(x) for x in normalized_layers]
                                trace_payload["raw_layer_rel_scores"] = raw_layer_rel_scores

                        self.last_trace.append(trace_payload)

                    image_mask[b, abs_idxs] = False
                    picked_embeds = inputs_embeds[b, abs_idxs, :]
                    ste_weight = (topk_scores - topk_scores.detach() + 1.0).unsqueeze(-1)
                    picked = picked_embeds * ste_weight
                    select_image_embeds.append(picked)

                #避免梯度传播
                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
            
                # --- 【新增检查点 1】 ---
                if torch.isnan(select_image_embeds).any() or torch.isinf(select_image_embeds).any():
                    print(f"🚨 警告: 在 pass_idx {pass_idx}，魔改5生成的 select_image_embeds 中发现了 NaN 或 Inf!")
                    print(f"最大值: {select_image_embeds.max().item()}, 最小值: {select_image_embeds.min().item()}")
                    #import pdb; pdb.set_trace()
                # -----------------------
            
                inputs_embeds_detached = inputs_embeds.detach().clone()
                for b in range(B):
                    if len(latent_lists[b]) > pass_idx:
                        #在特定位置用新的hidden_states替换原来的token embeddings。
                        t_idx = latent_lists[b][pass_idx]
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        #在指定位置t_idx用新的hidden_states替换原有的embedding,进而修改input embedding
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]

                inputs_embeds.data = inputs_embeds_detached
                new_inputs_embeds = []
                new_attention_mask = []
                new_position_ids = []
                new_original_mask = []
                new_image_mask = []
                batch_max_len = 0

                for b in range(B):
                    end_b = end
                    prefix_b = inputs_embeds[b, :end_b, :]    # (end_b, D) # 截取图片插入点之前的向量
                    suffix_b = inputs_embeds[b, end_b:, :]    # (old_len - end_b, D)  # 截取图片插入点之后的向量
                    v_embed_b = select_image_embeds[b]       # (K, D)  # 提取的图片向量
                    merged_b = torch.cat([prefix_b, v_embed_b, suffix_b], dim=0)  # (old_len+K, D) # 拼接：前缀 + 图片 + 后缀
                    new_inputs_embeds.append(merged_b)

                    # attention_mask
                    att_pref = attention_mask[b, :end_b]      # (end_b,)
                    att_suf  = attention_mask[b, end_b:]      # (old_len-end_b,)
                    # 为图片生成全为 1 的 mask，表示模型需要关注这些图像内容
                    att_v    = torch.ones(self.num_selected_patches, device=attention_mask.device, dtype=attention_mask.dtype)
                    merged_att = torch.cat([att_pref, att_v, att_suf], dim=0)  # (new_len,)
                    new_attention_mask.append(merged_att)

                    # position_ids 简单地重新生成了一串从 0 到 new_len-1 的连续整数作为新的位置索引。
                    new_pos = torch.arange(merged_b.size(0), device=position_ids.device)
                    new_position_ids.append(new_pos)

                    # original_mask 新增的图片token处mask为0
                    orig_pref = original_mask[b, :end_b]       # (end_b,)
                    orig_suf  = original_mask[b, end_b:]       # (old_len-end_b,)
                    orig_v    = torch.zeros(self.num_selected_patches, device=input_ids.device, dtype=torch.bool)
                    merged_orig = torch.cat([orig_pref, orig_v, orig_suf], dim=0)
                    new_original_mask.append(merged_orig)

                    # image_mask 新增的图片token处mask为0
                    img_pref = image_mask[b, :end_b]
                    img_suf  = image_mask[b, end_b:]
                    img_v    = torch.zeros(self.num_selected_patches, device=input_ids.device, dtype=torch.bool)
                    merged_img = torch.cat([img_pref, img_v, img_suf], dim=0)
                    new_image_mask.append(merged_img)

                    batch_max_len = max(batch_max_len, merged_b.size(0))

                #将循环处理得到的列表（List）数据重新封装，恢复成 Tensor 格式的 Batch（批次）数据，以便模型能够并行计算
                padded_embeds = []
                padded_att   = []
                padded_pos   = []
                padded_orig  = []
                padded_img   = []

                for b in range(B):
                    emb_b = new_inputs_embeds[b]
                    att_b = new_attention_mask[b]
                    pos_b = new_position_ids[b]
                    orig_b = new_original_mask[b]
                    img_b = new_image_mask[b]

                    padded_embeds.append(emb_b.unsqueeze(0))
                    padded_att.append(att_b.unsqueeze(0))
                    padded_pos.append(pos_b.unsqueeze(0))
                    padded_orig.append(orig_b.unsqueeze(0))
                    padded_img.append(img_b.unsqueeze(0))

                inputs_embeds = torch.cat(padded_embeds, dim=0)    
                attention_mask = torch.cat(padded_att, dim=0)      
                position_ids    = torch.cat(padded_pos, dim=0)     
                original_mask  = torch.cat(padded_orig, dim=0)
                image_mask     = torch.cat(padded_img, dim=0)   # (B, new_S)
                K = self.num_selected_patches
                # 当你把 $K$ 个图像特征（Patches）插入到原始文本序列中后，原本排在插入点之后的那些“特殊位置”或“潜在特征点”（Latent Positions）的索引就全部对不上了。这段逻辑就是在做索引重映射（Index Shifting）。
                for b in range(B):
                    for i, pos in enumerate(latent_lists[b]):
                        if pos > end:
                            latent_lists[b][i] = pos + K
                            logging.debug(f"latent pos: {latent_lists[b][i]}")

                if pass_idx + 1 >= max_n_latents:
                    end = inputs_embeds.size(1)
                else:
                    end = end + 1 + K

            #处理完多模态序列（拼接了图像和文本）后，正式调用**底层的语言模型（Base Causal LM）**进行前向传播的过程（处理完全部的laten token后的forward）
            if kv_cache:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            all_logits.append(outputs.logits)

        else:
            #应该是不使用latent的情况下（max为0）
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=False,
            )
            all_logits.append(outputs.logits)

        logits = torch.cat(all_logits, dim=-2)  # (B, total_len, V)
        B, final_S, V = logits.size()

        # 由于模型在 forward 过程中通过 torch.cat 插入了 $K$ 个图像 Patch，原始的 labels 长度已经与输出的 logits 长度不匹配了。
        # final_S 是拼接图像后的总长度
        new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype)
        for b in range(B):
            num_labels = labels.size(1)
            #将原始的 labels（即你希望模型预测的文本部分）填入 new_labels 的末尾
            new_labels[:, -num_labels:] = labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        
        # --- 【新增检查点 3】 ---
        if torch.isnan(shift_logits).any():
            print("🚨 警告: 最终拼接的 shift_logits 中包含 NaN！")
            #import pdb; pdb.set_trace()
        # -----------------------
        
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # --- 【最终检查】 ---
        if torch.isnan(loss):
            print("🚨 训练崩溃: 计算出的 Loss 为 NaN！优化器将跳过更新。")
            #import pdb; pdb.set_trace()
        # -------------------
        
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)


    def train(self, mode=True):
        self.base_causallm.train(mode)

    def eval(self):
        self.base_causallm.eval()
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            image_grid_thw: torch.Tensor = None,
            past_key_values: tuple = None,
            attention_mask: torch.Tensor = None,
            inputs_embeds: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            **kwargs
        ):
        
        self.base_causallm.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        pixel_values,
        image_grid_thw,
        max_new_tokens=16,
        output_embedding=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        eos_pos = None
        if self.enable_trace:
            self.last_trace = []

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        
        current_ids = input_ids.clone()

        position_ids = torch.arange(
            0, current_ids.shape[1], 
            dtype=torch.long, 
            device=current_ids.device
        ).reshape(1, -1)

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            labels=current_ids.clone(),  
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )


        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
            

        current_inputs_embeds = outputs.inputs_embeds  # shape: (1, seq_len_after_insertion, hidden_dim)
        current_seq_len = current_inputs_embeds.shape[1]
        

        current_attention_mask = torch.ones((1, current_seq_len), device=current_inputs_embeds.device)
        

        next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

        self.gen_forward_cnt += 1
        

        past_key_values = None
        

        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                logging.debug(f"no kv_cache, using full embedding sequence")
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(
                        0, current_inputs_embeds.shape[1], 
                    dtype=torch.long, 
                        device=current_inputs_embeds.device
                ).reshape(1, -1)
            else:
                logging.debug(f"using kv_cache, input_shape: {next_token_embedding.shape}")
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor([[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device)

            outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None, 
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            
            next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

            self.gen_forward_cnt += 1

            if self.gen_forward_cnt % 10 == 0 and self.gen_forward_cnt >= 10:
                logging.debug(f"gen_forward_cnt: {self.gen_forward_cnt}")

            if next_token == self.eos_token_id:
                logging.debug(f"EOS token encountered at position {len(tokens)}, stopping generation")
                break

        print("generate 315")
        
        
        if output_embedding:
            return torch.tensor(tokens).view(1, -1), current_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)


