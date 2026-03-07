# from transformers import AutoTokenizer, AutoProcessor
# from qwen_ivtlr import IVTLR  
# from transformers import Qwen2VLForConditionalGeneration
# import torch
# from torch_npu.contrib import transfer_to_npu
# import deepspeed
# from peft import LoraConfig,get_peft_model
# from qwen_vl_utils import process_vision_info
# from datasets import load_dataset
# import re
# import logging
# import json
# import os
# import time
# from datetime import timedelta
# logging.basicConfig(
#     filename='qwenvl_32_infer_time.log',
#     level=logging.DEBUG,
#     format='[%(asctime)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# import pdb

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def load_inference_model(checkpoint_path):
#     processor = AutoProcessor.from_pretrained("/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct")
#     tokenizer = AutoTokenizer.from_pretrained(
#         "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct",
#         use_fast=False,
#         trust_remote_code=True,
#         padding_side="right"
#     )
    
#     tokenizer.add_special_tokens({
#         "additional_special_tokens": [
#             "<|start-latent|>",
#             "<|end-latent|>",
#             "<|latent|>"
#         ]
#     })
    
#     base_model = Qwen2VLForConditionalGeneration.from_pretrained(
#         "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct",
#         device_map="cuda",
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         attn_implementation="eager"
#     )
#     base_model.resize_token_embeddings(len(tokenizer))
#     processor.tokenizer = tokenizer

#     lora_config = LoraConfig(
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         r=64,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         bias="none",
#         inference_mode=False
#     )
#     base_model = get_peft_model(base_model, lora_config)
    
#     latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
#     start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
#     end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
#     image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
#     visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
#     visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
#     model = IVTLR(
#         base_model,
#         latent_token_id=latent_id,
#         start_latent_id=start_id,
#         end_latent_id=end_id,
#         eos_token_id=tokenizer.eos_token_id,
#         image_token_id=image_token_id,
#         visual_start_id=visual_start_id, 
#         visual_end_id=visual_end_id
#     )
    
#     state_dict = torch.load(checkpoint_path, map_location="cpu")
#     print(state_dict.keys())
#     if any(k.startswith("module.") for k in state_dict.keys()):
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
#     model.load_state_dict(state_dict, strict=True)
#     print(model)
#     print("Successfully load")
    
#     model = model.to(device)
#     model.eval()
#     return model, processor, tokenizer

# model, processor, tokenizer = load_inference_model("/home/ma-user/work/lbx/IVT-LR/qwen_vl/output/m3cot_IVTLR/epoch_16_full_model_fp32.pth")

# os.makedirs("output", exist_ok=True)

# def format_prompt(example):
#     question = example["question"].strip()
#     rationale = example["rationale"].replace("\n", " ").strip()
#     answer = example["answer"].strip()
#     choices = example["choices"]
#     image = example["image"]

#     choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
#     user_prompt = (
#         f"[Question]:{{{question}}}\n"
#         f"[Options]:\n{choices_str}\n"
#         f"Answer:"
#     )
#     return user_prompt, rationale, answer, image

# def process_func(example):
#     prompt, rationale, answer, image = format_prompt(example)

#     return {
#         "question_raw": prompt,
#         "image_raw": image,
#         "gt_answer": answer,
#         "id": example["id"],
#         "choices": example["choices"],
#         "domain": example["domain"],
#         "topic": example["topic"]
#     }

# dataset = load_dataset("LightChen2333/M3CoT")
# val_dataset = dataset["test"]
# val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)

# def evaluate_and_save(eval_dataset, model, processor):
#     model.eval()
#     correct = 0
#     total = 0
#     total_generated_tokens = 0 
#     total_generate_time = 0.0  
    
#     output_path = "output/qwen2vl_32.jsonl"
#     with open(output_path, "a", encoding="utf-8") as f_out:
#         for ex in eval_dataset:
#             input_text = ex["question_raw"]
#             messages = [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
#                     {"type": "text", "text": input_text}
#                 ]
#             }]
#             text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             text = text + "<|latent|>" + "<|latent|>" + "<|latent|>"
#             image_inputs, video_inputs = process_vision_info(messages)
#             inputs = processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt"
#             ).to(device)
#             input_ids = inputs["input_ids"]
#             prompt_length = input_ids.shape[1]
            
#             generate_start_time = time.time()
#             with torch.no_grad():
#                 outputs = model.generate(
#                     input_ids=torch.tensor(inputs["input_ids"]), 
#                     attention_mask=torch.tensor(inputs["attention_mask"]),
#                     pixel_values=torch.tensor(inputs["pixel_values"]),
#                     image_grid_thw=torch.tensor(inputs["image_grid_thw"]),
#                     max_new_tokens=512
#                 )
#             generate_end_time = time.time()
#             sample_generate_time = generate_end_time - generate_start_time
#             total_generate_time += sample_generate_time
                        
#             generated_tokens = outputs[0, prompt_length:]
#             new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
#             output_text = processor.decode(outputs[0], skip_special_tokens=True)
#             logging.debug(f"[OUTPUT] {output_text}")
            
#             num_generated_tokens = len(generated_tokens)
#             total_generated_tokens += num_generated_tokens

#             cleaned_text = re.sub(
#                 r'(?<=answer:)\s*(\n+\s*)?assistant\b',
#                 '',
#                 output_text,
#                 flags=re.IGNORECASE
#             )
#             matches = re.finditer(
#                 r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
#                 cleaned_text,
#                 flags=re.IGNORECASE | re.DOTALL
#             )
#             candidates = {match.group(1).upper() for match in matches}
#             gt_answer = ex["gt_answer"].strip().upper()

#             if gt_answer in candidates:
#                 correct += 1
#                 logging.debug(f"correct: True")
#             total += 1
#             logging.debug(f"[TOTAL] {total}")

#             # pdb.set_trace()
#             message_question = ex["question_raw"]
#             message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
#             message_question = message_question.split("Answer:")[0].strip()

#             result = {
#                 "id": ex["id"],
#                 "choices": ex["choices"],
#                 "answer": ex["gt_answer"],
#                 "domain": ex["domain"],
#                 "topic": ex["topic"],
#                 "messages": [
#                     message_question,
#                     new_generated_text
#                 ]
#             }
#             f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
#             f_out.flush()
            
#         avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
#         avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
#         logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
#         logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
#         logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
    
# evaluate_and_save(val_dataset, model, processor)





from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
from torch_npu.contrib import transfer_to_npu
import deepspeed
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import re
import logging
import json
import os
import time
import argparse  # [新增] 用于命令行参数解析
from datetime import timedelta
import yaml

# 配置日志
logging.basicConfig(
    filename='qwenvl_infer.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"

# [修改] 增加 checkpoint_path 参数，不再硬编码
from expert_runtime import build_expert_runtime_from_config
import re as _re


def _inject_teacher_dims_from_checkpoint(config_dict, state_dict):
    if not isinstance(config_dict, dict):
        return
    expert_cfg = config_dict.get("expert")
    if not isinstance(expert_cfg, dict):
        return
    if not bool(expert_cfg.get("enabled", False)):
        return

    teacher_dims = expert_cfg.get("teacher_dims")
    if not isinstance(teacher_dims, dict):
        teacher_dims = {}
        expert_cfg["teacher_dims"] = teacher_dims

    for k, v in state_dict.items():
        m = _re.match(r"^expert_projectors\.([a-z0-9_]+)\.weight$", k)
        if not m:
            continue
        expert = m.group(1)
        if hasattr(v, "shape") and len(v.shape) == 2:
            teacher_dims.setdefault(expert, int(v.shape[0]))


def load_inference_model(checkpoint_path, model_base_path, config_path: str = None):
    config_dict = None
    state_dict = None
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

    # Load checkpoint early so we can infer projector dims (teacher_dims) if YAML omitted them.
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if config_dict is not None:
        _inject_teacher_dims_from_checkpoint(config_dict, state_dict)

    processor = AutoProcessor.from_pretrained(model_base_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_base_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|start-latent|>",
            "<|end-latent|>",
            "<|latent|>"
        ]
    })

    # Expert runtime is optional; when enabled, expert tokens are added as normal tokens
    # so that decode(skip_special_tokens=True) will keep them visible.
    expert_runtime = None
    if config_dict is not None:
        expert_runtime = build_expert_runtime_from_config(
            config_dict,
            tokenizer,
            add_tokens=True,
            for_inference=True,
        )
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_base_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    processor.tokenizer = tokenizer

    lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],  # 🌟【必须新增这一行】
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False
        )
    base_model = get_peft_model(base_model, lora_config)
    
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    visual_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    visual_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    model = IVTLR(
        base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        image_token_id=image_token_id,
        visual_start_id=visual_start_id, 
        visual_end_id=visual_end_id,
        model_path=model_base_path,
        expert_runtime=expert_runtime,
    )
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print("[Load Error] Failed to load checkpoint strictly.")
        print("If this checkpoint was trained with expert tokens/modules, pass `--config` with the same YAML used in training.")
        raise
    # print(model) # 保持输出简洁，注释掉
    print("Successfully load model")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer

# ==========================================
# [新增] ScienceQA 相关逻辑 (参考 Chameleon 实现)
# ==========================================

def extract_answer_scienceqa(text):

    digit_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+(\d)',
        r'the\s+answer\s+is\s+(\d)',
        r'answer\s+is:?\s*(\d)',
    ]
    
    for pattern in digit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer_idx = int(match.group(1))
            logging.debug(f"Extracted answer (digit): {answer_idx}")
            return answer_idx
        
    letter_patterns = [
        r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])',
        r'the\s+answer\s+is\s+([A-Z])',
        r'answer\s+is:?\s*([A-Z])',
    ]
    
    for pattern in letter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            answer_idx = ord(letter) - ord('A')
            logging.debug(f"Extracted answer (letter): {letter} -> index {answer_idx}")
            return answer_idx
    
    logging.warning(f"No answer pattern found in text: {text[:200]}")
    return -1

def format_prompt_scienceqa(example):
    """
    适配 ScienceQA 的 Prompt 格式
    """
    question = example["question"].strip()
    answer = example["answer"] 
    choices = example.get("choices", [])
    image = example["image"]
    
    # Qwen 通过 messages 处理 image，文本 prompt 中不需要 <image> token
    if choices:
        choices_str = "\n".join([f"({chr(65+i)}).{{{choice.strip()}}}" for i, choice in enumerate(choices)])
        user_prompt = (
            f"[Question]:{{{question}}}\n"
            f"[Options]:\n{choices_str}\n"
            f"Answer:"
        )
    else:
        user_prompt = f"[Question]:{{{question}}}\nAnswer:"
    
    return user_prompt, answer, image

def process_func_scienceqa(example, idx):
    prompt, answer, image = format_prompt_scienceqa(example)
    return {
        "idx": idx,
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
    }

def evaluate_scienceqa(model, processor, output_path="output/qwen2vl_scienceqa.jsonl"):
    print("Loading ScienceQA dataset...")
    dataset = load_dataset("derek-thomas/ScienceQA")
    val_dataset = dataset["test"]
    
    # 过滤包含图片的样本 (如果需要全量评测可注释此行，但代码逻辑假设有图片输入)
    val_dataset = val_dataset.filter(lambda e: e["image"] is not None)
    
    # 预处理
    val_dataset = val_dataset.map(lambda example, idx: process_func_scienceqa(example, idx), with_indices=True)

    print(f"Starting ScienceQA evaluation on {len(val_dataset)} samples...")
    
    model.eval()
    correct = 0
    total = 0
    results = {}
    total_generated_tokens = 0
    total_generate_time = 0.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ================= 【修改点 1：在循环外打开文件（追加模式）】 =================
    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in val_dataset:
            input_text = ex["question_raw"]

            # 构造 Qwen 格式的 messages
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                    {"type": "text", "text": input_text}
                ]
            }]

            # 应用 Chat Template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # 添加 latent tokens
            text = text + "<|latent|>" + "<|latent|>" + "<|latent|>"

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)

            prompt_length = inputs["input_ids"].shape[1]

            generate_start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                    max_new_tokens=512
                )
            generate_end_time = time.time()
            sample_generate_time = generate_end_time - generate_start_time
            total_generate_time += sample_generate_time

            generated_tokens = outputs[0, prompt_length:]
            new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            num_generated_tokens = len(generated_tokens)
            total_generated_tokens += num_generated_tokens

            # 提取答案
            pred_answer = extract_answer_scienceqa(new_generated_text)
            gt_answer = ex["gt_answer"]

            is_correct = (pred_answer == gt_answer)
            if is_correct:
                correct += 1

            total += 1

            # ================= 【修改点 2：单条记录构建并立即写入】 =================
            idx_str = str(ex["idx"])
            # 构建单条样本的结果字典
            single_result = {
                "id": idx_str,
                "pred": pred_answer,
                "gt": gt_answer,
                "correct": is_correct,
                "generated_text": new_generated_text
            }
            
            # 将单条结果转为 JSON 字符串，加上换行符，写入文件
            f_out.write(json.dumps(single_result, ensure_ascii=False) + "\n")
            # 强制刷新缓存，立刻写入磁盘！即使下一步崩溃，这条数据也保住了
            f_out.flush() 
            # =========================================================================
            
            # 记录结果
            idx_str = str(ex["idx"])
            results[idx_str] = {
                "pred": pred_answer,
                "gt": gt_answer,
                "correct": is_correct,
                "generated_text": new_generated_text
            }

            if total % 10 == 0:
                print(f"Processed {total}, Current Accuracy: {correct/total:.2%}")
                logging.info(f"[ScienceQA] Processed {total}, Accuracy: {correct/total:.2%}")

    final_acc = correct / total if total > 0 else 0
    avg_tokens = total_generated_tokens / total if total > 0 else 0
    avg_time = total_generate_time / total if total > 0 else 0
    
    print(f"\n[Final ScienceQA Results] Accuracy: {final_acc:.2%}, Avg Tokens: {avg_tokens:.1f}, Avg Time: {avg_time:.3f}s")
    
    # 🌟 将原本在这里的 output_data 构造和 json.dump(...) 代码全部删除 🌟
    # 我们可以把最终的汇总信息（准确率、平均耗时等）以一条特殊记录的形式追加到文件末尾
    with open(output_path, "a", encoding="utf-8") as f_out:
        summary_result = {
            "summary_type": "final_metrics",
            "accuracy": final_acc,
            "avg_tokens": avg_tokens,
            "avg_time": avg_time,
            "total_samples": total
        }
        f_out.write(json.dumps(summary_result, ensure_ascii=False) + "\n")


# ==========================================
# M3CoT 相关逻辑 (保持原有逻辑不变，封装为函数)
# ==========================================

def format_prompt_m3cot(example):
    question = example["question"].strip()
    rationale = example["rationale"].replace("\n", " ").strip()
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, rationale, answer, image

def process_func_m3cot(example):
    prompt, rationale, answer, image = format_prompt_m3cot(example)

    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }

def evaluate_m3cot(model, processor, output_path="output/qwen2vl_32.jsonl"):
    print("Loading M3CoT dataset...")
    dataset = load_dataset("LightChen2333/M3CoT")
    val_dataset = dataset["test"]
    val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func_m3cot)

    print(f"Starting M3CoT evaluation on {len(val_dataset)} samples...")
    
    model.eval()
    correct = 0
    total = 0
    total_generated_tokens = 0 
    total_generate_time = 0.0  
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in val_dataset:
            input_text = ex["question_raw"]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                    {"type": "text", "text": input_text}
                ]
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + "<|latent|>" + "<|latent|>" + "<|latent|>"
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(device)
            input_ids = inputs["input_ids"]
            prompt_length = input_ids.shape[1]
            
            generate_start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=torch.tensor(inputs["input_ids"]), 
                    attention_mask=torch.tensor(inputs["attention_mask"]),
                    pixel_values=torch.tensor(inputs["pixel_values"]),
                    image_grid_thw=torch.tensor(inputs["image_grid_thw"]),
                    max_new_tokens=512
                )
            generate_end_time = time.time()
            sample_generate_time = generate_end_time - generate_start_time
            total_generate_time += sample_generate_time
                        
            generated_tokens = outputs[0, prompt_length:]
            new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            output_text = processor.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"[OUTPUT] {output_text}")
            
            num_generated_tokens = len(generated_tokens)
            total_generated_tokens += num_generated_tokens

            # M3CoT 特有的答案匹配逻辑
            cleaned_text = re.sub(
                r'(?<=answer:)\s*(\n+\s*)?assistant\b',
                '',
                output_text,
                flags=re.IGNORECASE
            )
            matches = re.finditer(
                r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
                cleaned_text,
                flags=re.IGNORECASE | re.DOTALL
            )
            candidates = {match.group(1).upper() for match in matches}
            gt_answer = ex["gt_answer"].strip().upper()

            if gt_answer in candidates:
                correct += 1
                logging.debug(f"correct: True")
            total += 1
            logging.debug(f"[TOTAL] {total}")

            if total % 10 == 0:
                print(f"Processed {total}, Current M3CoT Accuracy: {correct/total:.2%}")

            message_question = ex["question_raw"]
            message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
            message_question = message_question.split("Answer:")[0].strip()

            result = {
                "id": ex["id"],
                "choices": ex["choices"],
                "answer": ex["gt_answer"],
                "domain": ex["domain"],
                "topic": ex["topic"],
                "messages": [
                    message_question,
                    new_generated_text
                ]
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()
            
        avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
        avg_time_per_sample = total_generate_time / total if total > 0 else 0
    
        logging.info(f"[FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
        logging.info(f"[FINAL] Total generate time: {total_generate_time:.2f}s ({timedelta(seconds=int(total_generate_time))})")
        logging.info(f"[FINAL] Avg generate time per sample: {avg_time_per_sample:.3f}s")
        # 建议在 evaluate_m3cot 函数末尾添加：
        print(f"Avg Tokens: {avg_generated_tokens:.1f}")
        print(f"Avg Time per Sample: {avg_time_per_sample:.3f}s")
        print(f"M3CoT Final Accuracy: {correct/total:.2%}")

# ==========================================
# 主程序入口 [新增/修改]   
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for IVT-LR (Qwen2-VL)")
    parser.add_argument("--checkpoint", type=str, default='/home/ma-user/work/lbx/IVT-LR/qwen_vl/output/m3cot_qwen2vl_2B_IVTLR/epoch_16_full_model_fp32.pth', help="Path to the model checkpoint (pth file)")
    parser.add_argument("--dataset", type=str, default="m3cot", choices=["m3cot", "scienceqa"], help="Dataset to evaluate (m3cot or scienceqa)")
    parser.add_argument("--model_base_path", type=str, default="/home/ma-user/work/lbx/models/Qwen2-VL-2B-Instruct", help="Path to the base Qwen2-VL model (2B or 7B)")
    parser.add_argument("--output_path", type=str, default="output/qwen2-vl-2b-m3cot.jsonl", help="Path to the base Qwen2-VL model (2B or 7B)")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config (enables expert tokens/modules if present)")
    
    args = parser.parse_args()
    output_path = args.output_path
    
    # 1. 加载模型
    model, processor, tokenizer = load_inference_model(args.checkpoint, args.model_base_path, config_path=args.config)

    
    # 2. 根据参数选择评测数据集
    if args.dataset == "m3cot":
        evaluate_m3cot(model, processor,output_path)
    elif args.dataset == "scienceqa":
        evaluate_scienceqa(model, processor,output_path)
    else:
        print("Invalid dataset selected.")









