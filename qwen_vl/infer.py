from transformers import AutoTokenizer, AutoProcessor
from qwen_ivtlr import IVTLR  
from transformers import Qwen2VLForConditionalGeneration
import torch
from torch_npu.contrib import transfer_to_npu
import deepspeed
from peft import LoraConfig,get_peft_model
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import re
import logging
import json
import os
import time
from datetime import timedelta
logging.basicConfig(
    filename='qwenvl_32_infer_time.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_inference_model(checkpoint_path):
    processor = AutoProcessor.from_pretrained("/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct",
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
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct",
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
        visual_end_id=visual_end_id
    )
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(state_dict.keys())
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
    print(model)
    print("Successfully load")
    
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer

model, processor, tokenizer = load_inference_model("/home/ma-user/work/lbx/IVT-LR/qwen_vl/output/m3cot_IVTLR/epoch_16_full_model_fp32.pth")

os.makedirs("output", exist_ok=True)

def format_prompt(example):
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

def process_func(example):
    prompt, rationale, answer, image = format_prompt(example)

    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "choices": example["choices"],
        "domain": example["domain"],
        "topic": example["topic"]
    }

dataset = load_dataset("LightChen2333/M3CoT")
val_dataset = dataset["test"]
val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)

def evaluate_and_save(eval_dataset, model, processor):
    model.eval()
    correct = 0
    total = 0
    total_generated_tokens = 0 
    total_generate_time = 0.0  
    
    output_path = "output/qwen2vl_32.jsonl"
    with open(output_path, "a", encoding="utf-8") as f_out:
        for ex in eval_dataset:
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

            # pdb.set_trace()
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
    
evaluate_and_save(val_dataset, model, processor)














# import torch
# import torch.multiprocessing as mp
# from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
# from qwen_ivtlr import IVTLR
# from peft import LoraConfig, get_peft_model
# from qwen_vl_utils import process_vision_info
# from datasets import load_dataset
# import re
# import logging
# import json
# import os
# import time
# from datetime import timedelta
# import glob

# # 确保 deepspeed 或其他库不干扰多进程
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def setup_logging(rank):
#     """为每个进程设置独立的日志文件"""
#     logging.basicConfig(
#         filename=f'qwenvl_32_infer_time_rank_{rank}.log',
#         level=logging.DEBUG,
#         format=f'[Rank {rank}] [%(asctime)s] %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S',
#         force=True # 强制重新配置 logging
#     )

# def load_inference_model(checkpoint_path, device_id):
#     model_path = "/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct"
    
#     processor = AutoProcessor.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
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
    
#     # 关键修改：指定 device_map 到具体的 GPU ID
#     base_model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_path,
#         device_map={"":"cuda:" + str(device_id)}, # 强制映射到指定显卡
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
#     if any(k.startswith("module.") for k in state_dict.keys()):
#         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
#     model.load_state_dict(state_dict, strict=True)
#     # 移动到指定 GPU
#     device = torch.device(f"cuda:{device_id}")
#     model = model.to(device)
#     model.eval()
    
#     return model, processor, tokenizer

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

# def worker_process(rank, world_size, checkpoint_path, output_dir):
#     """
#     单个 GPU 的推理工作进程
#     """
#     # 1. 设置设备和日志
#     setup_logging(rank)
#     device_id = rank
#     torch.cuda.set_device(device_id)
#     device = torch.device(f"cuda:{device_id}")
    
#     print(f"Process {rank}/{world_size} started on device {device}")

#     # 2. 加载模型
#     try:
#         model, processor, tokenizer = load_inference_model(checkpoint_path, device_id)
#         logging.info(f"Model loaded on rank {rank}")
#     except Exception as e:
#         logging.error(f"Failed to load model on rank {rank}: {e}")
#         return

#     # 3. 加载并分片数据集
#     # 注意：建议在 worker 内部加载数据集以避免 pickle 问题，或者由外部传入预处理好的 dataset
#     dataset = load_dataset("LightChen2333/M3CoT")
#     val_dataset = dataset["test"]
#     val_dataset = val_dataset.filter(lambda e: e["image"] is not None).map(process_func)
    
#     # 将数据集分片
#     sharded_dataset = val_dataset.shard(num_shards=world_size, index=rank)
#     logging.info(f"Rank {rank} processing {len(sharded_dataset)} samples")

#     # 4. 推理循环
#     temp_output_path = os.path.join(output_dir, f"qwen2vl_32_rank_{rank}.jsonl")
    
#     correct = 0
#     total = 0
#     total_generated_tokens = 0 
#     total_generate_time = 0.0  
    
#     with open(temp_output_path, "w", encoding="utf-8") as f_out:
#         for ex in sharded_dataset:
#             try:
#                 input_text = ex["question_raw"]
#                 messages = [{
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
#                         {"type": "text", "text": input_text}
#                     ]
#                 }]
#                 text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#                 text = text + "<|latent|>" + "<|latent|>" + "<|latent|>"
#                 image_inputs, video_inputs = process_vision_info(messages)
                
#                 inputs = processor(
#                     text=[text],
#                     images=image_inputs,
#                     videos=video_inputs,
#                     padding=True,
#                     return_tensors="pt"
#                 ).to(device)
                
#                 input_ids = inputs["input_ids"]
#                 prompt_length = input_ids.shape[1]
                
#                 generate_start_time = time.time()
#                 with torch.no_grad():
#                     # 确保 inputs 中的张量都在正确的 device 上 (processor output usually on CPU until .to())
#                     outputs = model.generate(
#                         input_ids=inputs["input_ids"], 
#                         attention_mask=inputs["attention_mask"],
#                         pixel_values=inputs["pixel_values"],
#                         image_grid_thw=inputs["image_grid_thw"],
#                         max_new_tokens=512
#                     )
#                 generate_end_time = time.time()
                
#                 sample_generate_time = generate_end_time - generate_start_time
#                 total_generate_time += sample_generate_time
                        
#                 generated_tokens = outputs[0, prompt_length:]
#                 new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
#                 output_text = processor.decode(outputs[0], skip_special_tokens=True)
#                 logging.debug(f"[OUTPUT] {output_text}")
                
#                 num_generated_tokens = len(generated_tokens)
#                 total_generated_tokens += num_generated_tokens

#                 cleaned_text = re.sub(
#                     r'(?<=answer:)\s*(\n+\s*)?assistant\b',
#                     '',
#                     output_text,
#                     flags=re.IGNORECASE
#                 )
#                 matches = re.finditer(
#                     r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
#                     cleaned_text,
#                     flags=re.IGNORECASE | re.DOTALL
#                 )
#                 candidates = {match.group(1).upper() for match in matches}
#                 gt_answer = ex["gt_answer"].strip().upper()

#                 if gt_answer in candidates:
#                     correct += 1
                
#                 total += 1

#                 message_question = ex["question_raw"]
#                 message_question = message_question.replace("<image>", "", 1).replace("Answer:", "", 1).strip()
#                 message_question = message_question.split("Answer:")[0].strip()

#                 result = {
#                     "id": ex["id"],
#                     "choices": ex["choices"],
#                     "answer": ex["gt_answer"],
#                     "domain": ex["domain"],
#                     "topic": ex["topic"],
#                     "messages": [
#                         message_question,
#                         new_generated_text
#                     ]
#                 }
#                 f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
#                 f_out.flush()
                
#             except Exception as e:
#                 logging.error(f"Error processing sample {ex.get('id', 'unknown')}: {e}")
#                 continue

#     avg_generated_tokens = total_generated_tokens / total if total > 0 else 0
#     avg_time_per_sample = total_generate_time / total if total > 0 else 0

#     logging.info(f"[Rank {rank} FINAL] Total: {total}, Correct: {correct}")
#     logging.info(f"[Rank {rank} FINAL] Avg generated tokens per sample: {avg_generated_tokens:.1f}")
#     logging.info(f"[Rank {rank} FINAL] Total generate time: {total_generate_time:.2f}s")
#     print(f"Rank {rank} finished. Processed {total} samples.")

# def merge_output_files(output_dir, final_filename, world_size):
#     """将所有 rank 的输出文件合并为一个"""
#     final_path = os.path.join(output_dir, final_filename)
#     print(f"Merging files into {final_path}...")
    
#     with open(final_path, "w", encoding="utf-8") as f_final:
#         for rank in range(world_size):
#             rank_file = os.path.join(output_dir, f"qwen2vl_32_rank_{rank}.jsonl")
#             if os.path.exists(rank_file):
#                 with open(rank_file, "r", encoding="utf-8") as f_rank:
#                     for line in f_rank:
#                         f_final.write(line)
#                 # 可选：合并后删除临时文件
#                 # os.remove(rank_file) 
#             else:
#                 print(f"Warning: Output file for rank {rank} not found.")
#     print("Merge complete.")

# if __name__ == "__main__":
#     # 配置路径
#     CHECKPOINT_PATH = "/home/ma-user/work/lbx/IVT-LR/qwen_vl/output/m3cot_IVTLR/epoch_4_full_model_fp32.pth"
#     OUTPUT_DIR = "output"
#     FINAL_FILENAME = "qwen2vl_32.jsonl"
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # 获取 GPU 数量
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available.")
    
#     world_size = torch.cuda.device_count()
#     print(f"Found {world_size} GPUs. Starting inference...")

#     # 启动多进程
#     mp.spawn(
#         worker_process,
#         args=(world_size, CHECKPOINT_PATH, OUTPUT_DIR),
#         nprocs=world_size,
#         join=True
#     )
    
#     # 汇总结果
#     merge_output_files(OUTPUT_DIR, FINAL_FILENAME, world_size)
#     print("All tasks finished.")

