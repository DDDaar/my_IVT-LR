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
        num_selected_patches: int = 32,
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

        ####################################################################

        #1. 获取 num_heads
        # 大多数 HF 模型（Qwen2-VL, Chameleon, Llama）使用 .num_attention_heads
        # GPT-2 使用 .n_head
        if hasattr(self.base_causallm.config, "num_attention_heads"):
            num_heads = self.base_causallm.config.num_attention_heads
        elif hasattr(self.base_causallm.config, "n_head"):
            num_heads = self.base_causallm.config.n_head
        else:
            raise ValueError("Cannot find number of attention heads in model config")


        if hasattr(self.base_causallm.config, "hidden_size"):
            hidden_size = self.base_causallm.config.hidden_size
        elif hasattr(self.base_causallm.config, "n_embd"):
            hidden_size = self.base_causallm.config.n_embd
            
        
        self.head_gate = nn.Sequential(
            nn.Linear(hidden_size, num_heads), # 输入 latent hidden state，输出每个 head 的权重
            nn.Softmax(dim=-1) # 保证权重和为 1
        )
        ####################################################################
        
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
                hidden_states = outputs.hidden_states[-1]     # 最后一层的隐藏状态 
                
                all_hidden_states = outputs.hidden_states  # 所有层的隐藏状态
                
                
                
                attentions    = outputs.attentions        #所有注意力层的注意力权重列表 list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)

#######################################################################原始                
#                 #   Top-K
#                 avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len) 将所有层的注意力矩阵在 heads（头）维度上拼接，(B, L * heads, seq_len, seq_len)---->(B, seq_len, seq_len)
#                 current_seq_len = avg_attn.size(1) #seq长度


                
################################################################################
                # # 使用刚加的模块：mlp选择
                # # --- [修改开始] 使用 head_fusion 层进行融合 ---
                
                # # 1. 对每一层的注意力矩阵应用 head_fusion
                # layer_fused_attns = []
                # for layer_attn in attentions:
                #     # layer_attn shape: (B, num_heads, S, S)
                    
                #     # 调整维度，将 num_heads 放到最后，以便 Linear 层处理
                #     # permute -> (B, S, S, num_heads)
                #     layer_attn_perm = layer_attn.permute(0, 2, 3, 1)
                    
                #     # 应用你的 head_fusion (Linear + Sigmoid)
                #     # 输入: (..., num_heads) -> 输出: (..., 1)
                #     # 结果 shape: (B, S, S, 1)
                #     fused_score = self.head_fusion(layer_attn_perm)
                    
                #     # 去掉最后一维 -> (B, S, S)
                #     layer_fused_attns.append(fused_score.squeeze(-1))

                # # 2. 将各个层融合 (这里采用层间平均)
                # # stack -> (num_layers, B, S, S)
                # # mean(dim=0) -> (B, S, S)
                # avg_attn = torch.stack(layer_fused_attns, dim=0).mean(dim=0)
                
                # # --- [修改结束] ---

                # current_seq_len = avg_attn.size(1)
################################################################################


################################################################################
## 使用hiddenstate->head提取各个head的权重进行加权求和
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
################################################################################
                

    

    
################################################################################
# 开始：逐层独立的 Head 门控融合，hiddenstate->head
                
                layer_fused_attns = []
                
                for layer_idx, layer_attn in enumerate(attentions):
                    # 获取当前层【输入】的隐藏状态 (即上一层的输出) 作为当前层门控的输入
                    # 你也可以使用 all_hidden_states[layer_idx + 1] 作为当前层【输出】的隐藏状态
                    current_layer_latent = all_hidden_states[layer_idx][:, end-1, :] # (B, Hidden_Size)
                    
                    # 使用当前层的隐藏状态，生成当前层专属的各个 Head 权重
                    dynamic_head_weights = self.head_gate(current_layer_latent) # (B, Num_Heads)
                    dynamic_head_weights = dynamic_head_weights.unsqueeze(-1).unsqueeze(-1) # (B, Num_Heads, 1, 1)
                    
                    # 加权求和: Sum(Attention * Weight) -> (B, S, S)
                    weighted_attn = (layer_attn * dynamic_head_weights).sum(dim=1)
                    layer_fused_attns.append(weighted_attn)
                
                # 层间融合：依然使用最基础的平均方式
                avg_attn = torch.stack(layer_fused_attns, dim=0).mean(dim=0)
                current_seq_len = avg_attn.size(1)

################################################################################
    
                select_image_embeds = []

                for b in range(B):
                    #最后一个位置的注意力图
                    last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                    vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    scores = last_attn.clone()
                    
                    allowed_positions = image_mask[b, :current_seq_len]  # shape=(S,)
                    invalid = ~allowed_positions
                    #将非图像位置的分数设为负无穷，确保不会被选中
                    scores[invalid] = float("-inf")

                    rel_scores = scores[vs+1 : ve]  # (image_len,)
                    #选择图像token中的topk个
                    topk_rel = rel_scores.topk(self.num_selected_patches, sorted=False)[1]  # rel idx
                    abs_idxs = (vs + 1) + topk_rel
                    logging.debug(f"topk_rel: {topk_rel}")
                    logging.debug(f"abs idx: {abs_idxs}")
                    image_mask[b, abs_idxs] = False

                    #提取对应位置的embedding
                    picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    select_image_embeds.append(picked)

                #避免梯度传播
                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
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
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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


