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

# ================= 答案提取逻辑 (统一使用新逻辑) =================

def extract_answer_universal(text):
    """
    统一的答案提取函数。
    策略：寻找文本中出现的 "Answer: X" 模式，并取最后一个。
    """
    # 匹配 Answer: A 或 Answer: (A) 或 Answer: Option A
    # re.IGNORECASE 忽略大小写
    # pattern 解释:
    #   Answer:      匹配固定的前缀
    #   \s* 匹配0个或多个空格
    #   (?:Option)?  非捕获组，可选匹配 "Option" 单词
    #   \s* 匹配空格
    #   \(?          可选匹配左括号
    #   ([A-Z])      捕获组：匹配单个大写字母 (A-Z) -> 这是我们要的答案
    #   \)?          可选匹配右括号
    pattern = r'Answer:\s*(?:Option)?\s*\(?([A-D])\)?'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if matches:
        # 返回最后一个匹配到的答案，通常这是模型总结的最终结论
        return matches[-1].upper()
    
    # 如果没找到标准格式，尝试回退逻辑：找最后出现的单独字母
    # 这是一个兜底策略，匹配行首或行尾的单独字母
    fallback_matches = re.findall(r'(?:^|\s)([A-D])(?:\.|,|$)', text)
    if fallback_matches:
        return fallback_matches[-1].upper()

    return "FAILED"

# ================= M3CoT 数据处理 =================

def format_prompt_m3cot(example):
    question = example["question"].strip()
    answer = example["answer"].strip()
    choices = example["choices"]
    image = example["image"]

    choices_str = "\n".join([f"{chr(65+i)}. {choice.strip()}" for i, choice in enumerate(choices)])
    
    # 【修改点】: 明确要求输出格式 Answer: X
    user_prompt = (
        f"[Question]: {question}\n"
        f"[Options]:\n{choices_str}\n\n"
        f"At the end of your response, output the final choice specifically in this format: Answer: X"
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

# ================= ScienceQA 数据处理 =================

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

    choices_str = "\n".join([
        f"{chr(65 + i)}. {choice.strip()}"   
        for i, choice in enumerate(choices)
    ])
    
    # 【修改点】: 明确要求输出格式 Answer: X
    user_prompt = (
        f"[Question]: {question}\n"
        f"[Options]:\n{choices_str}\n\n"
        f"At the end of your response, output the final choice specifically in this format: Answer: X"
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

# ================= 通用评测逻辑 =================

def evaluate_dataset(dataset_name, eval_dataset, model, processor, output_file):
    print(f"\n{'='*40}")
    print(f"Starting evaluation for {dataset_name} (Size: {len(eval_dataset)})")
    print(f"{'='*40}")
    
    correct = 0
    total = 0
    total_generate_time = 0.0
    
    # 清空或创建文件
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
            
            # 【修改点】: 使用统一的提取函数
            pred_answer = extract_answer_universal(new_generated_text)
            
            # 比对
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
    parser.add_argument("--dataset", type=str, default="all", choices=["m3cot", "scienceqa", "all"], help="Dataset to evaluate")
    parser.add_argument("--output_dir", type=str, default="output_ori", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if '2B' in args.model_path:
        args.output_dir = "output_ori_2B"    
    elif '7B' in args.model_path:
        args.output_dir = "output_ori_7B"
    else:
        pass
    
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
            # 打印详细错误堆栈以便调试
            import traceback
            traceback.print_exc()

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
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()