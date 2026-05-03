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
from collections import defaultdict

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

def normalize_vstar_label(label):
    if isinstance(label, int):
        if label < 0:
            return ""
        return chr(ord("A") + label)
    if isinstance(label, str):
        value = label.strip().upper()
        if not value:
            return ""
        if len(value) == 1 and "A" <= value <= "Z":
            return value
        digit_match = re.search(r"\d+", value)
        if digit_match:
            idx = int(digit_match.group(0))
            if idx >= 0:
                return chr(ord("A") + idx)
        letter_match = re.search(r"[A-Z]", value)
        if letter_match:
            return letter_match.group(0)
    return ""

def extract_option_set_from_text(text):
    if not isinstance(text, str):
        return None
    patterns = [
        r"^\s*\(?([A-Z])\)\s+",
        r"^\s*([A-Z])[\.\:]\s+",
        r"\n\s*\(?([A-Z])\)\s+",
        r"\n\s*([A-Z])[\.\:]\s+",
    ]
    options = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            options.add(match.group(1).upper())
    if options:
        return options
    return None

def extract_answer_vstar(text, valid_options=None):
    if not isinstance(text, str):
        return "FAILED"
    allowed = valid_options if valid_options else {chr(ord("A") + i) for i in range(10)}
    allowed = set(x.upper() for x in allowed)

    patterns = [
        r"(?:final\s+answer|the\s+answer\s+is|answer)\s*[:：]?\s*\(?([A-Z])\)?",
        r"(?:option|choice)\s*[:：]?\s*\(?([A-Z])\)?",
    ]

    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            letter = m.group(1).upper()
            if letter in allowed:
                matches.append(letter)
    if matches:
        return matches[-1]

    fallback_tokens = re.findall(r"\b([A-Z])\b", text.upper())
    for token in reversed(fallback_tokens):
        if token in allowed:
            return token
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

def format_prompt_vstar_bench(example):
    prompt = example.get("text", "")
    if not isinstance(prompt, str):
        prompt = str(prompt)
    prompt = prompt.strip()
    if not re.search(r"answer\s*[:：]?\s*$", prompt, re.IGNORECASE):
        prompt += "\nAt the end of your response, output the final choice specifically in this format: Answer: X"

    answer = normalize_vstar_label(example.get("label", ""))
    image = example.get("image")
    category = example.get("category", "unknown")
    question_id = example.get("question_id", "")
    valid_options = extract_option_set_from_text(example.get("text", ""))
    return prompt, answer, image, category, question_id, valid_options

def process_func_vstar(example, idx):
    prompt, answer, image, category, question_id, valid_options = format_prompt_vstar_bench(example)
    sample_id = question_id if question_id != "" else str(idx)
    return {
        "question_raw": prompt,
        "image_raw": image,
        "gt_answer": answer,
        "id": sample_id,
        "dataset": "vstar_bench",
        "category": category,
        "question_id": question_id,
        "valid_options": sorted(list(valid_options)) if valid_options else None,
    }

# ================= 通用评测逻辑 =================

def evaluate_dataset(dataset_name, eval_dataset, model, processor, output_file):
    print(f"\n{'='*40}")
    print(f"Starting evaluation for {dataset_name} (Size: {len(eval_dataset)})")
    print(f"{'='*40}")

    correct = 0
    total = 0
    total_generate_time = 0.0
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

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
            if dataset_name == "vstar_bench":
                valid_options = set(ex["valid_options"]) if ex.get("valid_options") else None
                pred_answer = extract_answer_vstar(new_generated_text, valid_options=valid_options)
            else:
                pred_answer = extract_answer_universal(new_generated_text)

            is_correct = (pred_answer == gt_answer)
            if is_correct:
                correct += 1
            total += 1

            if dataset_name == "vstar_bench":
                category = ex.get("category", "unknown")
                category_stats[category]["total"] += 1
                if is_correct:
                    category_stats[category]["correct"] += 1

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
            if dataset_name == "vstar_bench":
                result["category"] = ex.get("category", "unknown")
                result["question_id"] = ex.get("question_id", "")

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

        if dataset_name == "vstar_bench":
            print("  Category Accuracy:")
            for category in sorted(category_stats.keys()):
                c_total = category_stats[category]["total"]
                c_correct = category_stats[category]["correct"]
                c_acc = c_correct / c_total if c_total > 0 else 0.0
                print(f"    - {category}: {c_acc:.2%} ({c_correct}/{c_total})")
    else:
        print(f"[{dataset_name}] No samples processed.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate original Qwen2-VL on M3CoT, ScienceQA and VSTAR-Bench")
    parser.add_argument("--model_path", type=str, default="/home/ma-user/work/lbx/models/Qwen2-VL-7B-Instruct", help="Path to the model")
    parser.add_argument("--dataset", type=str, default="all", choices=["m3cot", "scienceqa", "vstar_bench", "all"], help="Dataset to evaluate")
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


    # 4. 评测 VSTAR-Bench
    if args.dataset in ["vstar_bench"]:
        try:
            print("Loading VSTAR-Bench dataset...")
            dataset = load_dataset("lmms-lab/vstar-bench")
            if "test" in dataset:
                test_dataset = dataset["test"]
            else:
                first_split = list(dataset.keys())[0]
                test_dataset = dataset[first_split]
                print(f"[Warning] 'test' split not found, fallback to split: {first_split}")
            test_dataset = test_dataset.filter(lambda e: e["image"] is not None).map(process_func_vstar, with_indices=True)

            output_file = os.path.join(args.output_dir, "vstar_bench_qwen2vl_ori_results.jsonl")
            evaluate_dataset("vstar_bench", test_dataset, model, processor, output_file)
        except Exception as e:
            print(f"Failed to evaluate VSTAR-Bench: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()