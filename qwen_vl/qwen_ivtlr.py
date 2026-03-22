import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
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
        num_selected_patches: int = 8,
        model_path: str = None,  # [新增参数]
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
            
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        # self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")

        # Cache resolved visual tower to avoid repeated wrapper traversal.
        self._cached_visual_tower = None
        self._cached_visual_tower_path = None

    def _get_visual_tower(self):
        if self._cached_visual_tower is not None:
            return self._cached_visual_tower

        # Traverse common wrapper stacks for DDP/Accelerate/PEFT/HF wrappers.
        queue = [("base_causallm", self.base_causallm)]
        visited = set()
        traversed_paths = []

        while queue:
            cur_path, cur_obj = queue.pop(0)
            if cur_obj is None:
                continue
            obj_id = id(cur_obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if hasattr(cur_obj, "visual"):
                self._cached_visual_tower = getattr(cur_obj, "visual")
                self._cached_visual_tower_path = f"{cur_path}.visual"
                if self._cached_visual_tower_path not in (
                    "base_causallm.visual",
                    "base_causallm.model.visual",
                ):
                    print(f"[IVTLR] Visual tower resolved via `{self._cached_visual_tower_path}`")
                return self._cached_visual_tower

            for attr in ("module", "model", "base_model"):
                nxt = getattr(cur_obj, attr, None)
                if nxt is not None:
                    nxt_path = f"{cur_path}.{attr}"
                    queue.append((nxt_path, nxt))
                    if len(traversed_paths) < 16:
                        traversed_paths.append(nxt_path)

        raise AttributeError(
            "Cannot find vision tower on base_causallm wrapper chain. "
            f"Tried wrapper paths={traversed_paths}"
        )

    def _get_visual_dtype(self):
        visual_tower = self._get_visual_tower()
        if hasattr(visual_tower, "get_dtype"):
            return visual_tower.get_dtype()
        first_param = next(visual_tower.parameters(), None)
        if first_param is not None:
            return first_param.dtype
        return torch.float32

    def _normalize_visual_candidate(self, x):
        if not torch.is_tensor(x):
            return None
        if x.dim() == 3:
            return x.reshape(-1, x.shape[-1])
        if x.dim() == 2:
            return x
        return None

    def _extract_visual_hidden(self, visual_outputs, visual_tower, expected_tokens=None):
        candidates = []

        def _add_candidate(name, tensor):
            normalized = self._normalize_visual_candidate(tensor)
            if normalized is not None:
                candidates.append((name, normalized))

        # 1) Direct outputs
        if torch.is_tensor(visual_outputs):
            _add_candidate("tensor", visual_outputs)
        elif isinstance(visual_outputs, (tuple, list)):
            for idx, item in enumerate(visual_outputs):
                _add_candidate(f"tuple[{idx}]", item)
        else:
            # 2) ModelOutput-like attributes
            for attr in ("last_hidden_state", "pooler_output", "image_embeds"):
                if hasattr(visual_outputs, attr):
                    _add_candidate(attr, getattr(visual_outputs, attr))
            if hasattr(visual_outputs, "hidden_states") and visual_outputs.hidden_states is not None:
                hs = visual_outputs.hidden_states
                if isinstance(hs, (tuple, list)) and len(hs) > 0:
                    _add_candidate("hidden_states[-1]", hs[-1])

        if len(candidates) == 0:
            raise TypeError(f"Unsupported visual output type: {type(visual_outputs)}")

        if expected_tokens is None:
            return candidates[0][1]

        # Prefer exact shape match first.
        for _, cand in candidates:
            if cand.shape[0] == expected_tokens:
                return cand

        # Some variants return pre-merger features. Try visual merger if available.
        if hasattr(visual_tower, "merger"):
            for _, cand in candidates:
                try:
                    merged = visual_tower.merger(cand)
                    merged = self._normalize_visual_candidate(merged)
                    if merged is not None and merged.shape[0] == expected_tokens:
                        return merged
                except Exception:
                    continue

        candidate_shapes = [f"{name}:{tuple(t.shape)}" for name, t in candidates]
        raise ValueError(
            "Unable to align visual features with expected image tokens. "
            f"expected_tokens={expected_tokens}, candidates={candidate_shapes}"
        )

    def _run_visual_tower(self, pixel_values, image_grid_thw, expected_tokens=None):
        visual_tower = self._get_visual_tower()

        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided.")
        if image_grid_thw.dim() == 1:
            image_grid_thw = image_grid_thw.unsqueeze(0)

        pixel_values = pixel_values.to(self._get_visual_dtype())

        # Compatible call styles across Qwen2-VL implementations.
        try:
            visual_outputs = visual_tower(pixel_values, grid_thw=image_grid_thw)
        except TypeError:
            try:
                visual_outputs = visual_tower(pixel_values, image_grid_thw=image_grid_thw)
            except TypeError:
                visual_outputs = visual_tower(pixel_values, image_grid_thw)

        image_embeds = self._extract_visual_hidden(
            visual_outputs=visual_outputs,
            visual_tower=visual_tower,
            expected_tokens=expected_tokens,
        )
        return image_embeds, image_grid_thw
        
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
#         #魔改2、3
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
            
        
#         self.head_gate = nn.Sequential(
#             nn.Linear(hidden_size, num_heads), # 输入 latent hidden state，输出每个 head 的权重
#             nn.Softmax(dim=-1) # 保证权重和为 1
#         )
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
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            image_embeds, image_grid_thw = self._run_visual_tower(
                pixel_values,
                image_grid_thw,
                expected_tokens=n_image_tokens,
            )
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens={n_image_tokens}, features={image_embeds.shape[0]}, "
                    f"pixel_values_shape={tuple(pixel_values.shape)}, "
                    f"image_grid_thw_shape={tuple(image_grid_thw.shape)}"
                )
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
        max_len = 3000
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
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, start:end],
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
                attentions    = outputs.attentions        #所有注意力层的注意力权重列表 list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)

#######################################################################原始                
                #   Top-K
                # Streamed layer/head averaging to avoid allocating one giant
                # [B, layers*heads, S, S] temporary tensor.
                avg_attn = None
                for layer_attn in attentions:
                    layer_mean = layer_attn.mean(dim=1)  # (B, S, S)
                    if avg_attn is None:
                        avg_attn = layer_mean
                    else:
                        avg_attn = avg_attn + layer_mean
                avg_attn = avg_attn / max(len(attentions), 1)
                current_seq_len = avg_attn.size(1) #seq长度

                select_image_embeds = []

                for b in range(B):
                    #最后一个位置的注意力图
                    last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                    vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    rel_allowed = image_mask[b, vs + 1 : ve]  # shape=(image_len,)
                    rel_scores = last_attn[vs + 1 : ve].masked_fill(~rel_allowed, float("-inf"))
                    #选择图像token中的topk个
                    topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
                    abs_idxs = (vs + 1) + topk_rel
                    logging.debug(f"topk_rel: {topk_rel}")
                    logging.debug(f"abs idx: {abs_idxs}")
                    image_mask[b, abs_idxs] = False

                    #提取对应位置的embedding
                    picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    select_image_embeds.append(picked)
               #截止到'避免梯度传播'前面
                    
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
#                 # 在 forward 循环中修改 (约 175 行附近)
#                 # hidden_states: (B, Seq_Len, Hidden_Size)
#                 # 我们关注的是产生 Attention 的那个 Latent Token，即位置 end-1
#                 current_latent_vector = hidden_states[:, end-1, :] # (B, Hidden_Size)
                
#                 # 生成动态权重 (B, Num_Heads)
#                 dynamic_head_weights = self.head_gate(current_latent_vector) 
#                 dynamic_head_weights = dynamic_head_weights.unsqueeze(-1).unsqueeze(-1) # (B, Num_Heads, 1, 1)
                
#                 # 开始融合
#                 layer_fused_attns = []
#                 for layer_attn in attentions:
#                     # layer_attn: (B, Num_Heads, S, S)
#                     # 我们只需要最后一行 (Latent Token 对其他 Token 的关注度)
#                     # 注意：原始代码取了 avg_attn[b, end-1]，我们在融合前就可以切片以节省显存，或者在融合后切片
                    
#                     # 加权求和: Sum(Attention * Weight)
#                     # (B, Num_Heads, S, S) * (B, Num_Heads, 1, 1) -> Sum dim=1 -> (B, S, S)
#                     weighted_attn = (layer_attn * dynamic_head_weights).sum(dim=1)
#                     layer_fused_attns.append(weighted_attn)
                
#                 # 层间融合 (依然可以先用平均，或者参考方案二)
#                 avg_attn = torch.stack(layer_fused_attns, dim=0).mean(dim=0)
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

#                     # 【核心修改点】同时保留 topk_scores，以此作为梯度的桥梁
#                     topk_scores, topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False) 

#                     abs_idxs = (vs + 1) + topk_rel
#                     logging.debug(f"topk_rel: {topk_rel}")
#                     logging.debug(f"abs idx: {abs_idxs}")
#                     image_mask[b, abs_idxs] = False

#                     # #提取对应位置的embedding
#                     # picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
#                     # select_image_embeds.append(picked)

                    
#                     # 6. 从原始 inputs_embeds 中提取对应的 patch 特征
#                     picked_embeds = inputs_embeds[b, abs_idxs, :]  # (K, D)
#                     # 7. 【使用直通估计器 (STE) 融合梯度】
#                     # 正向传播：ste_weight 为 1.0，picked 特征值保持不变
#                     # 反向传播：梯度将通过 topk_scores 流向 rel_scores -> last_attn -> head_gate
#                     # 梯度会以 $1 \times \text{gradient}$ 的大小，完好无损地传递给 topk_scores，进而流向产生这个分数的 rel_scores 和 head_gate
#                     ste_weight = (topk_scores - topk_scores.detach() + 1.0).unsqueeze(-1)
#                     picked = picked_embeds * ste_weight
                    
#                     select_image_embeds.append(picked)
                
# ################################################################################
                

    

    
    
################################################################################
# # 开始：逐层独立的 Head 门控融合，hiddenstate->head，魔改3
                
                # layer_fused_attns = []
                
                # for layer_idx, layer_attn in enumerate(attentions):
                #     # 获取当前层【输入】的隐藏状态 (即上一层的输出) 作为当前层门控的输入
                #     # 你也可以使用 all_hidden_states[layer_idx + 1] 作为当前层【输出】的隐藏状态
                #     current_layer_latent = all_hidden_states[layer_idx][:, end-1, :] # (B, Hidden_Size)
                    
                #     # 使用当前层的隐藏状态，生成当前层专属的各个 Head 权重
                #     dynamic_head_weights = self.head_gate(current_layer_latent) # (B, Num_Heads)
                #     dynamic_head_weights = dynamic_head_weights.unsqueeze(-1).unsqueeze(-1) # (B, Num_Heads, 1, 1)
                    
                #     # 加权求和: Sum(Attention * Weight) -> (B, S, S)
                #     weighted_attn = (layer_attn * dynamic_head_weights).sum(dim=1)
                #     layer_fused_attns.append(weighted_attn)
                
                # #层间融合：依然使用最基础的平均方式
                # avg_attn = torch.stack(layer_fused_attns, dim=0).mean(dim=0)
                # current_seq_len = avg_attn.size(1)

                # select_image_embeds = []

                # for b in range(B):
                #     #最后一个位置的注意力图
                #     last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                #     vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                #     scores = last_attn.clone()
                    
                #     allowed_positions = image_mask[b, :current_seq_len]  # shape=(S,)
                #     invalid = ~allowed_positions
                #     #将非图像位置的分数设为负无穷，确保不会被选中
                #     scores[invalid] = float("-inf")

                #     rel_scores = scores[vs+1 : ve]  # (image_len,)
                #     #选择图像token中的topk个
                #     topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
                #     abs_idxs = (vs + 1) + topk_rel
                #     logging.debug(f"topk_rel: {topk_rel}")
                #     logging.debug(f"abs idx: {abs_idxs}")
                #     image_mask[b, abs_idxs] = False

                #     #提取对应位置的embedding
                #     picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                #     select_image_embeds.append(picked)

################################################################################
#                 # =====================================================================
#                 # 开始：使用可学习视觉选择器 (Cross-Attention) 选择 Top-K Patches
#                 # =====================================================================
                
# # 1. 提取当前 Latent Token 的隐藏状态作为 Query -> (B, 1, D)
#                 latent_hidden = hidden_states[:, end-1:end, :] 
#                 q = self.visual_q_proj(latent_hidden) # (B, 1, embed_dim)

#                 select_image_embeds = []

#                 for b in range(B):
#                     vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    
#                     # === 【修复点】：切片获取当前样本 b 的 Query ===
#                     q_b = q[b:b+1, :, :] # (1, 1, embed_dim)
                    
#                     # 2. 提取当前图片所有 Patch 的隐藏状态作为 Key -> (1, num_patches, D)
#                     image_hidden = hidden_states[b:b+1, vs+1:ve, :] 
#                     k = self.visual_k_proj(image_hidden) # (1, num_patches, embed_dim)
                    
#                     # === 【修复点】：使用 q_b 计算，并用 [0, 0] 安全降维 ===
#                     # q_b: (1, 1, embed_dim) | k.transpose: (1, embed_dim, num_patches)
#                     # bmm 结果为 (1, 1, num_patches)，取 [0, 0] 后变为一维的 (num_patches,)
#                     scores = torch.bmm(q_b, k.transpose(1, 2))[0, 0] / self.temperature
                    
#                     # 4. 屏蔽已经被之前的 Latent 选择过的 Patch
#                     allowed_positions = image_mask[b, vs+1:ve]
#                     scores[~allowed_positions] = float("-inf")

#                     # === 增加：将 Score 转为概率分布，建立梯度图 ===
#                     probs = torch.softmax(scores, dim=-1)

#                     # 5. 取 Top-K 及其对应的概率
#                     topk_probs, topk_rel = probs.topk(self.num_selected_patches, sorted=False)
#                     abs_idxs = (vs + 1) + topk_rel
                    
#                     logging.debug(f"topk_rel: {topk_rel}")
#                     logging.debug(f"abs idx: {abs_idxs}")
                    
#                     image_mask[b, abs_idxs] = False
                    
#                     # 6. 从原始 inputs_embeds 中提取对应的 patch 进行拼接
#                     picked_embeds = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    
#                     # === 增加：使用直通估计器 (STE) 融合梯度 ===
#                     # 正向传播时相当于 picked_embeds * 1.0，特征大小不变
#                     # 反向传播时，梯度会流向 topk_probs，从而更新 q_proj 和 k_proj
#                     ste_weight = (topk_probs - topk_probs.detach() + 1.0).unsqueeze(-1)
#                     picked = picked_embeds * ste_weight
                    
#                     select_image_embeds.append(picked)

#                 # =====================================================================
#                 # 结束：选择完毕，后面拼接 inputs_embeds_detached 等逻辑保持原有不变
#                 # =====================================================================
################################################################################



###################################################################### 
#                 #   魔改5，直接融合成新的视觉特征，而不是复用输入的embedding
#                 #   Top-K
#                 avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len) 将所有层的注意力矩阵在 heads（头）维度上拼接，(B, L * heads, seq_len, seq_len)---->(B, seq_len, seq_len)
#                 current_seq_len = avg_attn.size(1) #seq长度

#                 select_image_embeds = []

#                 for b in range(B):
#                     #最后一个位置的注意力图
#                     last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
#                     vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    
#                     # 1. 提取当前图像的所有深层隐藏状态作为 Key 和 Value
#                     image_hidden = hidden_states[b:b+1, vs+1:ve, :] # (1, num_patches, D)

#                     # 2. 将当前 Latent Token 的状态加到可学习的 Query 上，融入当前的“思考上下文”
#                     current_latent = hidden_states[b:b+1, end-1:end, :] # (1, 1, D)
#                     q = self.visual_queries + current_latent # (1, K, D)

#                     # 3. 通过 Cross-Attention 软提取图像特征
#                     # 这样不仅可微，而且模型学会了"该看哪里就融合哪里"，完全不需要 STE
#                     fused_visual_tokens, _ = self.cross_attn(query=q, key=image_hidden, value=image_hidden)

#                     # 4. 经过一个 MLP 并作为选出的图像 Embeddings
#                     picked = self.visual_proj(fused_visual_tokens).squeeze(0) # (K, D)
#                     select_image_embeds.append(picked)

#                     # 注意：使用这种方法，你不需要再去修改 image_mask 将选过的位置设为 False，
#                     # 因为每次都是对全局图像基于当前 Latent 进行条件查询。
#                     #截止到'避免梯度传播'前面
                    
######################################################################
                
    
    
    ################################################################################
# # 使用hiddenstate->head、hiddenstate->layer提取各个head的权重进行加权求和，魔改2plus
#                 # 在 forward 循环中修改 (约 175 行附近)
#                 # hidden_states: (B, Seq_Len, Hidden_Size)
#                 # 我们关注的是产生 Attention 的那个 Latent Token，即位置 end-1
#                 current_latent_vector = hidden_states[:, end-1, :] # (B, Hidden_Size)
                
#                 # 生成动态权重 (B, Num_Heads)
#                 dynamic_head_weights = self.head_gate(current_latent_vector) 
#                 dynamic_head_weights = dynamic_head_weights.unsqueeze(-1).unsqueeze(-1) # (B, Num_Heads, 1, 1)
                
#                 # 开始融合
#                 layer_fused_attns = []
#                 for layer_attn in attentions:
#                     # layer_attn: (B, Num_Heads, S, S)
#                     # 我们只需要最后一行 (Latent Token 对其他 Token 的关注度)
#                     # 注意：原始代码取了 avg_attn[b, end-1]，我们在融合前就可以切片以节省显存，或者在融合后切片
                    
#                     # 加权求和: Sum(Attention * Weight)
#                     # (B, Num_Heads, S, S) * (B, Num_Heads, 1, 1) -> Sum dim=1 -> (B, S, S)
#                     weighted_attn = (layer_attn * dynamic_head_weights).sum(dim=1)
#                     layer_fused_attns.append(weighted_attn)
                
#                 # 层间融合 (依然可以先用平均，或者参考方案二)
                
#                   # 改为（层权重也由 latent hidden state 动态生成）
#                 layer_weights = self.layer_gate(current_latent_vector)          # (B, num_layers)
#                 stacked = torch.stack(layer_fused_attns, dim=1)                 # (B, num_layers, S, S)
#                 avg_attn = (stacked * layer_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B, S, S)

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

#                     # 【核心修改点】同时保留 topk_scores，以此作为梯度的桥梁
#                     topk_scores, topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False) 

#                     abs_idxs = (vs + 1) + topk_rel
#                     logging.debug(f"topk_rel: {topk_rel}")
#                     logging.debug(f"abs idx: {abs_idxs}")
#                     image_mask[b, abs_idxs] = False

#                     # #提取对应位置的embedding
#                     # picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
#                     # select_image_embeds.append(picked)

                    
#                     # 6. 从原始 inputs_embeds 中提取对应的 patch 特征
#                     picked_embeds = inputs_embeds[b, abs_idxs, :]  # (K, D)
#                     # 7. 【使用直通估计器 (STE) 融合梯度】
#                     # 正向传播：ste_weight 为 1.0，picked 特征值保持不变
#                     # 反向传播：梯度将通过 topk_scores 流向 rel_scores -> last_attn -> head_gate
#                     # 梯度会以 $1 \times \text{gradient}$ 的大小，完好无损地传递给 topk_scores，进而流向产生这个分数的 rel_scores 和 head_gate
#                     ste_weight = (topk_scores - topk_scores.detach() + 1.0).unsqueeze(-1)
#                     picked = picked_embeds * ste_weight
                    
#                     select_image_embeds.append(picked)
                
# ################################################################################

                #避免梯度传播
                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
            
                # --- 【新增检查点 1】 ---
                if torch.isnan(select_image_embeds).any() or torch.isinf(select_image_embeds).any():
                    print(f"🚨 警告: 在 pass_idx {pass_idx}，魔改5生成的 select_image_embeds 中发现了 NaN 或 Inf!")
                    print(f"最大值: {select_image_embeds.max().item()}, 最小值: {select_image_embeds.min().item()}")
                    #import pdb; pdb.set_trace()
                # -----------------------
            
                inputs_embeds_detached = inputs_embeds.clone()
                for b in range(B):
                    if len(latent_lists[b]) > pass_idx:
                        #在特定位置用新的hidden_states替换原来的token embeddings。
                        t_idx = latent_lists[b][pass_idx]
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        #在指定位置t_idx用新的hidden_states替换原有的embedding,进而修改input embedding
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :].detach()

                inputs_embeds = inputs_embeds_detached
                K = self.num_selected_patches
                old_len = inputs_embeds.size(1)
                new_len = old_len + K
                hidden_dim = inputs_embeds.size(-1)

                # Pre-allocate merged tensors to avoid repeated torch.cat allocations.
                merged_inputs_embeds = inputs_embeds.new_empty((B, new_len, hidden_dim))
                merged_inputs_embeds[:, :end, :] = inputs_embeds[:, :end, :]
                merged_inputs_embeds[:, end : end + K, :] = select_image_embeds
                merged_inputs_embeds[:, end + K :, :] = inputs_embeds[:, end:, :]

                merged_attention_mask = attention_mask.new_empty((B, new_len))
                merged_attention_mask[:, :end] = attention_mask[:, :end]
                merged_attention_mask[:, end : end + K] = 1
                merged_attention_mask[:, end + K :] = attention_mask[:, end:]

                merged_position_ids = torch.arange(
                    new_len, device=position_ids.device, dtype=position_ids.dtype
                ).unsqueeze(0).expand(B, -1)

                merged_original_mask = original_mask.new_empty((B, new_len))
                merged_original_mask[:, :end] = original_mask[:, :end]
                merged_original_mask[:, end : end + K] = False
                merged_original_mask[:, end + K :] = original_mask[:, end:]

                image_mask_active = image_mask
                if image_mask_active.size(1) > old_len:
                    image_mask_active = image_mask_active[:, :old_len]
                elif image_mask_active.size(1) < old_len:
                    pad_cols = old_len - image_mask_active.size(1)
                    image_mask_active = torch.cat(
                        [
                            image_mask_active,
                            torch.zeros(
                                (B, pad_cols),
                                dtype=image_mask_active.dtype,
                                device=image_mask_active.device,
                            ),
                        ],
                        dim=1,
                    )

                merged_image_mask = image_mask_active.new_empty((B, new_len))
                merged_image_mask[:, :end] = image_mask_active[:, :end]
                merged_image_mask[:, end : end + K] = False
                merged_image_mask[:, end + K :] = image_mask_active[:, end:]

                inputs_embeds = merged_inputs_embeds
                attention_mask = merged_attention_mask
                position_ids = merged_position_ids
                original_mask = merged_original_mask
                image_mask = merged_image_mask   # (B, new_S)
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
                    output_hidden_states=True,
                    output_attentions=False,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
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
        loss = None
        if labels is not None:
            B, final_S, _ = logits.size()

            # 由于模型在 forward 过程中通过 torch.cat 插入了 $K$ 个图像 Patch，原始的 labels 长度已经与输出的 logits 长度不匹配了。
            # final_S 是拼接图像后的总长度
            new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype)
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
