# import torch
# from torch_npu.contrib import transfer_to_npu
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from datasets import load_dataset
# import re
# import logging
# import json
# import os
# import time
# from datetime import timedelta
# import argparse

# # 设置日志
# logging.basicConfig(
#     filename='infer_ori_qwen2vl.log',
#     level=logging.DEBUG,
#     format='[%(asctime)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def load_inference_model(model_path):
#     """
#     加载原版 Qwen2-VL 模型，不包含 IVTLR 包装器
#     """
#     print(f"Loading original Qwen2-VL model from {model_path}...")
    
#     # 加载 Processor 和 Tokenizer
#     processor = AutoProcessor.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path,
#         use_fast=False,
#         trust_remote_code=True,
#         padding_side="right"
#     )
    
#     # 直接加载 Qwen2VL 模型
#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         model_path,
#         device_map="cuda",
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         attn_implementation="eager"
#     )
    
#     # 确保 tokenizer 设置正确
#     processor.tokenizer = tokenizer
#     model.eval()
    
#     print("Model loaded successfully.")
#     return model, processor, tokenizer

# # ================= M3CoT 数据处理 =================

# def format_prompt_m3cot(example):
#     question = example["question"].strip()
#     # rationale = example["rationale"].replace("\n", " ").strip() # 推理时不需要 rationale
#     answer = example["answer"].strip()
#     choices = example["choices"]
#     image = example["image"]

#     choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
#     user_prompt = (
#         f"[Question]:{{{question}}}\n"
#         f"[Options]:\n{choices_str}\n"
#         f"Answer:"
#     )
#     return user_prompt, answer, image

# def process_func_m3cot(example):
#     prompt, answer, image = format_prompt_m3cot(example)
#     return {
#         "question_raw": prompt,
#         "image_raw": image,
#         "gt_answer": answer,
#         "id": example["id"],
#         "dataset": "m3cot"
#     }

# def extract_answer_m3cot(output_text):
#     # 复用 infer.py 中的 M3CoT 解析逻辑
#     cleaned_text = re.sub(
#         r'(?<=answer:)\s*(\n+\s*)?assistant\b',
#         '',
#         output_text,
#         flags=re.IGNORECASE
#     )
#     matches = re.finditer(
#         r'(?:the\s+answer\s+is|Answer:)\s*[\n\s]*([A-Z])',
#         cleaned_text,
#         flags=re.IGNORECASE | re.DOTALL
#     )
#     candidates = [match.group(1).upper() for match in matches]
#     if candidates:
#         return candidates[-1]
#     return "FAILED"

# # ================= ScienceQA 数据处理 =================

# def format_prompt_sqa(example):
#     question = example["question"].strip()
    
#     # 处理答案: SQA 的 answer 可能是 int 索引，需要转为 A/B/C
#     answer = example["answer"]
#     if isinstance(answer, int):
#         answer = chr(65 + answer)
#     else:
#         # 如果已经是字符串，尝试转换或保留
#         try:
#             # 尝试判断是否是数字字符串 '0', '1'
#             answer_int = int(answer)
#             answer = chr(65 + answer_int)
#         except:
#             answer = str(answer).strip().upper()
        
#     choices = example["choices"]
#     image = example["image"]

#     # 保持与 qwenvl_run.py 中训练逻辑一致的 Prompt 格式
#     choices_str = "[Options]:\n"+"\n".join([
#         f"({chr(65 + i)}).{{{choice.strip()}}}"  
#         for i, choice in enumerate(choices)
#     ])
    
#     question_with_braces = f"{{{question}}}"
    
#     user_prompt = (
#         f"[Question]:{question_with_braces}\n"
#         f"{choices_str}\n"
#         f"Answer:"
#     )
#     return user_prompt, answer, image

# def process_func_sqa(example, idx):
#     prompt, answer, image = format_prompt_sqa(example)
#     # SQA test 集有时没有 id 字段，使用索引代替
#     sample_id = example.get("id", str(idx))
#     return {
#         "question_raw": prompt,
#         "image_raw": image,
#         "gt_answer": answer,
#         "id": sample_id,
#         "dataset": "scienceqa"
#     }

# def extract_answer_sqa(text):
#     # 针对 ScienceQA 的解析逻辑，匹配 A/B/C/D...
#     patterns = [
#         r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])',
#         r'the\s+answer\s+is\s+([A-Z])',
#         r'answer\s+is:?\s*([A-Z])',
#         r'Answer:\s*([A-Z])',
#         r'^([A-Z])$' # 仅输出一个字母的情况
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             return match.group(1).upper()
    
#     # 如果没找到，尝试在最后几个字符里找选项
#     last_chars = text[-10:]
#     match = re.search(r'([A-Z])', last_chars)
#     if match:
#         return match.group(1).upper()
        
#     return "FAILED"


# # ================= 通用评测逻辑 =================

# def evaluate_dataset(dataset_name, eval_dataset, model, processor, output_file):
#     print(f"\n{'='*40}")
#     print(f"Starting evaluation for {dataset_name} (Size: {len(eval_dataset)})")
#     print(f"{'='*40}")
    
#     correct = 0
#     total = 0
#     total_generate_time = 0.0
    
#     # 清空或创建输出文件
#     with open(output_file, "w", encoding="utf-8") as f:
#         pass

#     for i, ex in enumerate(eval_dataset):
#         try:
#             input_text = ex["question_raw"]
            
#             # 构建对话消息
#             messages = [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
#                     {"type": "text", "text": input_text}
#                 ]
#             }]
            
#             # 应用 Chat Template
#             text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
#             # 这里的 text 就是纯文本 Prompt，不再添加 <|latent|>
            
#             # 处理视觉信息
#             image_inputs, video_inputs = process_vision_info(messages)
            
#             inputs = processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt"
#             ).to(device)
            
#             # 推理
#             prompt_length = inputs["input_ids"].shape[1]
#             generate_start_time = time.time()
            
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_new_tokens=512
#                 )
                
#             generate_end_time = time.time()
#             total_generate_time += (generate_end_time - generate_start_time)
            
#             # 解码
#             generated_tokens = outputs[0, prompt_length:]
#             output_full_text = processor.decode(outputs[0], skip_special_tokens=True)
#             new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            
#             logging.debug(f"[{dataset_name} ID:{ex['id']}] Generated: {new_generated_text}")
            
#             # 提取答案
#             gt_answer = ex["gt_answer"]
            
#             if dataset_name == "m3cot":
#                 # M3CoT 倾向于使用 output_full_text 进行解析（根据 infer.py 逻辑）
#                 # 但 infer.py 里的 cleaned_text 逻辑主要针对 assistant 输出
#                 # 这里我们传 full_text，并在解析函数里处理
#                 pred_answer = extract_answer_m3cot(output_full_text)
#             else:
#                 # ScienceQA 通常直接解析生成的新文本
#                 pred_answer = extract_answer_sqa(new_generated_text)
            
#             is_correct = (pred_answer == gt_answer)
#             if is_correct:
#                 correct += 1
#             total += 1
            
#             # 打印/保存结果
#             if total % 10 == 0:
#                 print(f"[{dataset_name}] Processed {total}/{len(eval_dataset)}. Acc: {correct/total:.2%}")
            
#             result = {
#                 "id": ex["id"],
#                 "gt_answer": gt_answer,
#                 "pred_answer": pred_answer,
#                 "correct": is_correct,
#                 "generated_text": new_generated_text,
#                 "full_output": output_full_text
#             }
            
#             with open(output_file, "a", encoding="utf-8") as f_out:
#                 f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
#         except Exception as e:
#             logging.error(f"Error processing sample {ex.get('id', 'unknown')}: {str(e)}")
#             print(f"Error on sample {ex.get('id', 'unknown')}: {e}")
#             continue

#     if total > 0:
#         avg_time = total_generate_time / total
#         acc = correct / total
#         print(f"\n[{dataset_name}] FINAL RESULTS:")
#         print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
#         print(f"  Avg Time: {avg_time:.4f}s")
#         print(f"  Results saved to: {output_file}")
#     else:
#         print(f"[{dataset_name}] No samples processed.")

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate original Qwen2-VL on M3CoT and ScienceQA")
#     parser.add_argument("--model_path", type=str, default="/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct", help="Path to the model")
#     parser.add_argument("--dataset", type=str, default="scienceqa", choices=["m3cot", "scienceqa", "all"], help="Dataset to evaluate")
#     parser.add_argument("--output_dir", type=str, default="output_ori", help="Output directory")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # 1. 加载模型
#     model, processor, tokenizer = load_inference_model(args.model_path)
    
#     # 2. 评测 M3CoT
#     if args.dataset in ["m3cot", "all"]:
#         try:
#             print("Loading M3CoT dataset...")
#             dataset = load_dataset("LightChen2333/M3CoT")
#             test_dataset = dataset["test"]
#             # 过滤无图数据并进行处理
#             test_dataset = test_dataset.filter(lambda e: e["image"] is not None).map(process_func_m3cot)
            
#             output_file = os.path.join(args.output_dir, "m3cot_qwen2vl_ori_results.jsonl")
#             evaluate_dataset("m3cot", test_dataset, model, processor, output_file)
#         except Exception as e:
#             print(f"Failed to evaluate M3CoT: {e}")

#     # 3. 评测 ScienceQA
#     if args.dataset in ["scienceqa", "all"]:
#         try:
#             print("Loading ScienceQA dataset...")
#             dataset = load_dataset("derek-thomas/ScienceQA")
#             test_dataset = dataset["test"]
#             # 过滤无图数据并进行处理，使用 with_indices 获取索引
#             test_dataset = test_dataset.filter(lambda e: e["image"] is not None).map(process_func_sqa, with_indices=True)
            
#             output_file = os.path.join(args.output_dir, "scienceqa_qwen2vl_ori_results.jsonl")
#             evaluate_dataset("scienceqa", test_dataset, model, processor, output_file)
#         except Exception as e:
#             print(f"Failed to evaluate ScienceQA: {e}")

# if __name__ == "__main__":
#     main()






import torch
from torch_npu.contrib import transfer_to_npu
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import re
import logging
import json
import os
import time
from datetime import timedelta
import argparse

# 设置日志
logging.basicConfig(
    filename='infer_ori_qwen2vl.log',
    level=logging.DEBUG,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_inference_model(model_path):
    """
    加载原版 Qwen2-VL 模型
    """
    print(f"Loading original Qwen2-VL model from {model_path}...")
    
    # 加载 Processor 和 Tokenizer
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 直接加载 Qwen2VL 模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    # 确保 tokenizer 设置正确
    processor.tokenizer = tokenizer
    model.eval()
    
    print("Model loaded successfully.")
    return model, processor, tokenizer

# ================= M3CoT 数据处理 (保持不变) =================

def format_prompt_m3cot(example):
    question = example["question"].strip()
    # rationale = example["rationale"].replace("\n", " ").strip() 
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    # M3CoT 格式: A.{选项}
    choices_str = "\n".join([f"{chr(65+i)}.{{{choice.strip()}}}" for i, choice in enumerate(choices)])
    user_prompt = (
        f"[Question]:{{{question}}}\n"
        f"[Options]:\n{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, answer, image

def process_func_m3cot(example):
    prompt, answer, image = format_prompt_m3cot(example)
    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": example["id"],
        "dataset": "m3cot"
    }

def extract_answer_m3cot(output_text):
    # 复用 infer.py 中的 M3CoT 解析逻辑
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
    candidates = [match.group(1).upper() for match in matches]
    if candidates:
        return candidates[-1]
    return "FAILED"

# ================= ScienceQA 数据处理 (已修改) =================

def format_prompt_sqa(example):
    question = example["question"].strip()
    
    # 处理答案: SQA 的 answer 可能是 int 索引，需要转为 A/B/C
    answer = example["answer"]
    if isinstance(answer, int):
        answer = chr(65 + answer)
    else:
        try:
            answer_int = int(answer)
            answer = chr(65 + answer_int)
        except:
            answer = str(answer).strip().upper()
        
    choices = example["choices"]
    image = example["image"]

    # --- 修改点 1：对齐 M3CoT 格式，移除选项外的括号 ---
    # 原代码: f"({chr(65 + i)}).{{{choice.strip()}}}"
    # 新代码: f"{chr(65 + i)}.{{{choice.strip()}}}"
    choices_str = "[Options]:\n"+"\n".join([
        f"{chr(65 + i)}.{{{choice.strip()}}}"  
        for i, choice in enumerate(choices)
    ])
    
    question_with_braces = f"{{{question}}}"
    
    user_prompt = (
        f"[Question]:{question_with_braces}\n"
        f"{choices_str}\n"
        f"Answer:"
    )
    return user_prompt, answer, image

def process_func_sqa(example, idx):
    prompt, answer, image = format_prompt_sqa(example)
    sample_id = example.get("id", str(idx))
    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": sample_id,
        "dataset": "scienceqa"
    }

def extract_answer_sqa(text):
    patterns = [
        # [新增] 专门匹配 "B.{false}" 这种直接复制选项格式的情况
        r'^([A-Z])\s*\.\s*\{',  
        # 原有逻辑
        r'Therefore,?\s*the\s+answer\s+is\s+([A-Z])',
        r'the\s+answer\s+is\s+([A-Z])',
        r'answer\s+is:?\s*([A-Z])',
        r'Answer:\s*([A-Z])',
        r'^([A-Z])$' 
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
    return "FAILED"

# ================= 通用评测逻辑 (保持不变) =================

def evaluate_dataset(dataset_name, eval_dataset, model, processor, output_file):
    print(f"\n{'='*40}")
    print(f"Starting evaluation for {dataset_name} (Size: {len(eval_dataset)})")
    print(f"{'='*40}")
    
    correct = 0
    total = 0
    total_generate_time = 0.0
    
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    for i, ex in enumerate(eval_dataset):
        try:
            input_text = ex["question_raw"]
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": ex["image_raw"], "resized_height": 280, "resized_width": 280},
                    {"type": "text", "text": input_text}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                    **inputs,
                    max_new_tokens=512
                )
                
            generate_end_time = time.time()
            total_generate_time += (generate_end_time - generate_start_time)
            
            generated_tokens = outputs[0, prompt_length:]
            output_full_text = processor.decode(outputs[0], skip_special_tokens=True)
            new_generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
            
            logging.debug(f"[{dataset_name} ID:{ex['id']}] Generated: {new_generated_text}")
            
            gt_answer = ex["gt_answer"]
            
            if dataset_name == "m3cot":
                pred_answer = extract_answer_m3cot(output_full_text)
            else:
                pred_answer = extract_answer_sqa(new_generated_text)
            
            is_correct = (pred_answer == gt_answer)
            if is_correct:
                correct += 1
            total += 1
            
            if total % 10 == 0:
                print(f"[{dataset_name}] Processed {total}/{len(eval_dataset)}. Acc: {correct/total:.2%}")
            
            result = {
                "id": ex["id"],
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "correct": is_correct,
                "generated_text": new_generated_text,
                "full_output": output_full_text
            }
            
            with open(output_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logging.error(f"Error processing sample {ex.get('id', 'unknown')}: {str(e)}")
            print(f"Error on sample {ex.get('id', 'unknown')}: {e}")
            continue

    if total > 0:
        avg_time = total_generate_time / total
        acc = correct / total
        print(f"\n[{dataset_name}] FINAL RESULTS:")
        print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
        print(f"  Avg Time: {avg_time:.4f}s")
        print(f"  Results saved to: {output_file}")
    else:
        print(f"[{dataset_name}] No samples processed.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate original Qwen2-VL on M3CoT and ScienceQA")
    parser.add_argument("--model_path", type=str, default="/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct", help="Path to the model")
    parser.add_argument("--dataset", type=str, default="scienceqa", choices=["m3cot", "scienceqa", "all"], help="Dataset to evaluate")
    parser.add_argument("--output_dir", type=str, default="output_ori", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    model, processor, tokenizer = load_inference_model(args.model_path)
    
    # 2. 评测 M3CoT
    if args.dataset in ["m3cot", "all"]:
        try:
            print("Loading M3CoT dataset...")
            dataset = load_dataset("LightChen2333/M3CoT")
            test_dataset = dataset["test"]
            test_dataset = test_dataset.filter(lambda e: e["image"] is not None).map(process_func_m3cot)
            
            output_file = os.path.join(args.output_dir, "m3cot_qwen2vl_ori_results.jsonl")
            evaluate_dataset("m3cot", test_dataset, model, processor, output_file)
        except Exception as e:
            print(f"Failed to evaluate M3CoT: {e}")

    # 3. 评测 ScienceQA
    if args.dataset in ["scienceqa", "all"]:
        try:
            print("Loading ScienceQA dataset...")
            dataset = load_dataset("derek-thomas/ScienceQA")
            test_dataset = dataset["test"]
            test_dataset = test_dataset.filter(lambda e: e["image"] is not None).map(process_func_sqa, with_indices=True)
            
            output_file = os.path.join(args.output_dir, "scienceqa_qwen2vl_ori_results.jsonl")
            evaluate_dataset("scienceqa", test_dataset, model, processor, output_file)
        except Exception as e:
            print(f"Failed to evaluate ScienceQA: {e}")

if __name__ == "__main__":
    main()